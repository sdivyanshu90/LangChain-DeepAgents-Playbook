from __future__ import annotations

import sqlite3
import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings


def validate_sql(sql: str) -> bool:
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        return False
    blocked = ["insert ", "update ", "delete ", "drop ", "alter ", ";"]
    return not any(token in normalized for token in blocked)


class DataState(TypedDict, total=False):
    db_path: str
    question: str
    query_plan: str
    sql: str
    sql_safe: bool
    rows: Annotated[list[tuple], operator.add]
    answer: str


def build_app(settings: Settings):
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You write a short data query plan for the given question."),
            ("human", "Question: {question}"),
        ]
    )
    sql_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You generate a single SQLite SELECT query from the question and plan. Return only SQL."),
            ("human", "Question: {question}\nPlan: {query_plan}"),
        ]
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You summarize query results concisely and stay grounded in the provided rows."),
            ("human", "Question: {question}\nSQL: {sql}\nRows: {rows}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    planner_chain = planner_prompt | model | StrOutputParser()
    sql_chain = sql_prompt | model | StrOutputParser()
    answer_chain = answer_prompt | model | StrOutputParser()

    def plan_node(state: DataState) -> DataState:
        return {"query_plan": planner_chain.invoke({"question": state["question"]})}

    def sql_node(state: DataState) -> DataState:
        sql = sql_chain.invoke({"question": state["question"], "query_plan": state["query_plan"]}).strip()
        return {"sql": sql}

    def validate_node(state: DataState) -> DataState:
        return {"sql_safe": validate_sql(state["sql"])}

    def execute_node(state: DataState) -> DataState:
        if not state["sql_safe"]:
            return {"rows": [], "answer": "The generated query was blocked by the SQL safety policy."}
        connection = sqlite3.connect(state["db_path"])
        try:
            cursor = connection.execute(state["sql"])
            rows = cursor.fetchmany(20)
        finally:
            connection.close()
        answer = answer_chain.invoke({"question": state["question"], "sql": state["sql"], "rows": rows})
        return {"rows": rows, "answer": answer}

    graph = StateGraph(DataState)
    graph.add_node("plan", plan_node)
    graph.add_node("sql", sql_node)
    graph.add_node("validate", validate_node)
    graph.add_node("execute", execute_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "sql")
    graph.add_edge("sql", "validate")
    graph.add_edge("validate", "execute")
    graph.add_edge("execute", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
