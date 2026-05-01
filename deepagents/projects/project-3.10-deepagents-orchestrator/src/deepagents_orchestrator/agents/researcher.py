from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ..config import Settings


class ResearcherState(TypedDict, total=False):
    goal: str
    last_instruction: str
    task_plan: Annotated[list[str], operator.add]
    completed_work: Annotated[list[str], operator.add]


def build_researcher_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a research specialist. Gather evidence and next-step ideas for the goal."),
            ("human", "Goal: {goal}\nInstruction: {last_instruction}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = prompt | model | StrOutputParser()

    def search_node(state: ResearcherState) -> ResearcherState:
        result = chain.invoke(
            {
                "goal": state["goal"],
                "last_instruction": state.get("last_instruction", "Start with research."),
            }
        )
        return {"task_plan": [result], "completed_work": [f"Researcher: {result}"]}

    graph = StateGraph(ResearcherState)
    graph.add_node("search", search_node)
    graph.add_edge(START, "search")
    graph.add_edge("search", END)
    return graph.compile()