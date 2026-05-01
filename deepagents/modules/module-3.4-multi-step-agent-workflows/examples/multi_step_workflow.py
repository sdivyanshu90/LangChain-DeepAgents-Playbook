from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict, total=False):
    task: str
    retries: int
    step_status: str
    result: str


def execute_step(state: WorkflowState) -> WorkflowState:
    retries = state.get("retries", 0)
    if retries == 0:
        return {"retries": retries + 1, "step_status": "retry"}
    return {"retries": retries, "step_status": "success", "result": f"Completed task: {state['task']}"}


def recover(state: WorkflowState) -> WorkflowState:
    return {"result": f"Task blocked after retries: {state['task']}"}


def route(state: WorkflowState) -> str:
    if state.get("step_status") == "success":
        return "done"
    if state.get("retries", 0) >= 1:
        return "retry_once"
    return "recover"


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("execute_step", execute_step)
    graph.add_node("recover", recover)
    graph.add_edge(START, "execute_step")
    graph.add_conditional_edges(
        "execute_step",
        route,
        {"done": END, "retry_once": "execute_step", "recover": "recover"},
    )
    graph.add_edge("recover", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"task": "refresh deployment status", "retries": 0})
    print(result)


if __name__ == "__main__":
    main()
