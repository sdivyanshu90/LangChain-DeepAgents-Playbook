from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ObservabilityState(TypedDict, total=False):
    task: str
    events: list[str]
    status: str


def start_node(state: ObservabilityState) -> ObservabilityState:
    events = list(state.get("events", []))
    events.append("start_node: accepted task")
    return {"events": events, "status": "running"}


def decide_node(state: ObservabilityState) -> ObservabilityState:
    events = list(state.get("events", []))
    events.append("decide_node: route=finish")
    return {"events": events, "status": "finished"}


def build_graph():
    graph = StateGraph(ObservabilityState)
    graph.add_node("start_node", start_node)
    graph.add_node("decide_node", decide_node)
    graph.add_edge(START, "start_node")
    graph.add_edge("start_node", "decide_node")
    graph.add_edge("decide_node", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"task": "triage a customer issue", "events": []})
    print(result)


if __name__ == "__main__":
    main()
