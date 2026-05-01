from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from .agents import build_analyst_app, build_researcher_app, build_reviewer_app, build_writer_app
from .config import Settings
from .supervisor import build_supervisor_node


class OrchestratorState(TypedDict, total=False):
    goal: str
    task_plan: Annotated[list[str], operator.add]
    completed_work: Annotated[list[str], operator.add]
    draft: str
    review_feedback: dict
    revision_count: int
    final_deliverable: str
    last_instruction: str


def route_after_review(state: OrchestratorState) -> str:
    latest_review = state.get("review_feedback", {})
    if latest_review.get("score", 10) >= 8 or state.get("revision_count", 0) >= 3:
        return "supervisor"
    return "writer"


def build_app(settings: Settings):
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", build_supervisor_node(settings))
    graph.add_node("researcher", build_researcher_app(settings))
    graph.add_node("analyst", build_analyst_app(settings))
    graph.add_node("writer", build_writer_app(settings))
    graph.add_node("reviewer", build_reviewer_app(settings))
    graph.add_edge(START, "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_conditional_edges("reviewer", route_after_review, {"supervisor": "supervisor", "writer": "writer"})
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
