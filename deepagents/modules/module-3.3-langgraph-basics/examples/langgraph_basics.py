from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class RequestState(TypedDict, total=False):
    request: str
    route: Literal["fast_path", "review_path"]
    result: str


def triage(state: RequestState) -> RequestState:
    request = state["request"].lower()
    route = "review_path" if "approval" in request else "fast_path"
    return {"route": route}


def fast_path(state: RequestState) -> RequestState:
    return {"result": f"Handled quickly: {state['request']}"}


def review_path(state: RequestState) -> RequestState:
    return {"result": f"Sent for review: {state['request']}"}


def build_graph():
    graph = StateGraph(RequestState)
    graph.add_node("triage", triage)
    graph.add_node("fast_path", fast_path)
    graph.add_node("review_path", review_path)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", lambda state: state["route"], {"fast_path": "fast_path", "review_path": "review_path"})
    graph.add_edge("fast_path", END)
    graph.add_edge("review_path", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"request": "Please request approval for a vendor exception."})
    print(result)


if __name__ == "__main__":
    main()
