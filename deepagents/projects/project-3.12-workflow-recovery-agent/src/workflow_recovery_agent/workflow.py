from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings


class RecoveryState(TypedDict, total=False):
    task: str
    retries: int
    max_retries: int
    checkpoints: Annotated[list[str], operator.add]
    partial_results: Annotated[list[str], operator.add]
    status: str
    result: str
    fallback_used: bool


def execute_once(state: RecoveryState) -> RecoveryState:
    retries = state.get("retries", 0)
    if retries < 1:
        return {
            "retries": retries + 1,
            "checkpoints": [f"attempt {retries + 1}: primary step failed"],
            "partial_results": [f"attempt {retries + 1}: captured failure context for '{state['task']}'"],
            "status": "retry",
            "fallback_used": False,
        }
    return {
        "retries": retries + 1,
        "checkpoints": [f"attempt {retries + 1}: primary step succeeded"],
        "partial_results": [f"attempt {retries + 1}: primary step recovered for '{state['task']}'"],
        "status": "success",
        "result": f"Recovered workflow output for task: {state['task']}",
        "fallback_used": False,
    }


def build_fallback_chain(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "The primary workflow failed after {max_retries} retries. "
                "Produce the best possible partial result from what is available.",
            ),
            (
                "human",
                "Task: {task}\nPartial results: {partial_results}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    return prompt | model | StrOutputParser()


def fallback(state: RecoveryState, *, fallback_chain, max_retries: int) -> RecoveryState:
    partial_results = "\n".join(state.get("partial_results", [])) or "none"
    result = fallback_chain.invoke(
        {
            "max_retries": max_retries,
            "task": state["task"],
            "partial_results": partial_results,
        }
    )
    return {
        "checkpoints": ["fallback: returned degraded but safe result"],
        "status": "fallback",
        "result": result,
        "fallback_used": True,
    }


def build_app(settings: Settings):
    fallback_chain = build_fallback_chain(settings)

    def route(state: RecoveryState) -> str:
        if state.get("status") == "success":
            return "done"
        if state.get("retries", 0) >= state.get("max_retries", settings.max_retries):
            return "fallback"
        return "retry"

    def fallback_node(state: RecoveryState) -> RecoveryState:
        return fallback(state, fallback_chain=fallback_chain, max_retries=state.get("max_retries", settings.max_retries))

    graph = StateGraph(RecoveryState)
    graph.add_node("execute", execute_once)
    graph.add_node("fallback", fallback_node)
    graph.add_edge(START, "execute")
    graph.add_conditional_edges("execute", route, {"done": END, "retry": "execute", "fallback": "fallback"})
    graph.add_edge("fallback", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
