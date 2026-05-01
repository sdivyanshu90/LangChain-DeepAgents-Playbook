from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DeepAgentState(TypedDict, total=False):
    goal: str
    plan: list[str]
    work_log: list[str]
    should_continue: bool
    final_output: str


def planner(state: DeepAgentState) -> DeepAgentState:
    return {"plan": ["search sources", "read evidence", "draft summary"], "work_log": []}


def executor(state: DeepAgentState) -> DeepAgentState:
    work_log = list(state.get("work_log", []))
    remaining = [item for item in state.get("plan", []) if item not in work_log]
    if remaining:
        work_log.append(remaining[0])
    return {"work_log": work_log}


def reflector(state: DeepAgentState) -> DeepAgentState:
    should_continue = len(state.get("work_log", [])) < len(state.get("plan", []))
    final_output = "" if should_continue else f"Completed plan for: {state['goal']}"
    return {"should_continue": should_continue, "final_output": final_output}


def route(state: DeepAgentState) -> str:
    return "continue" if state.get("should_continue", False) else "finish"


def build_graph():
    graph = StateGraph(DeepAgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("reflector", reflector)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reflector")
    graph.add_conditional_edges("reflector", route, {"continue": "executor", "finish": END})
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"goal": "Build a short research brief about launch readiness."})
    print(result)


if __name__ == "__main__":
    main()
