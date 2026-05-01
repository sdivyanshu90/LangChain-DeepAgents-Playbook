from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ..config import Settings


class AnalystState(TypedDict, total=False):
    goal: str
    last_instruction: str
    task_plan: Annotated[list[str], operator.add]
    completed_work: Annotated[list[str], operator.add]


def build_analyst_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an analyst. Turn research notes into a sharper execution perspective."),
            (
                "human",
                "Goal: {goal}\nInstruction: {last_instruction}\nResearch notes:\n{task_plan}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = prompt | model | StrOutputParser()

    def analyze_node(state: AnalystState) -> AnalystState:
        result = chain.invoke(
            {
                "goal": state["goal"],
                "last_instruction": state.get("last_instruction", "Analyze the current material."),
                "task_plan": "\n".join(state.get("task_plan", [])),
            }
        )
        return {"completed_work": [f"Analyst: {result}"]}

    graph = StateGraph(AnalystState)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()