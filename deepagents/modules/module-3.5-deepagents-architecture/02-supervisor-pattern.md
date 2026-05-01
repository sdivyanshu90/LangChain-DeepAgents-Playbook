[← What Makes an Agent Deep](01-what-makes-an-agent-deep.md) | [Next → Swarm Handoff](03-swarm-handoff-pattern.md)

---

# 02 — Supervisor Pattern

## Why a Single Routing Point?

When multiple specialist agents exist, the naive approach is to let each agent decide
who should handle the next step. The result is a peer-to-peer conversation that quickly
becomes a cycle: Agent A hands off to Agent B, which hands back to Agent A, which...

The Supervisor pattern solves this by introducing a single routing authority: one node
that receives the goal, decides which specialist should act next, and collects results.
No specialist can route to another specialist directly — they can only return to the
Supervisor.

---

## Real-World Analogy

An air traffic controller (Supervisor) manages the runway. Pilots (specialists) don't
talk to each other directly — they all communicate through the controller. This prevents
conflicts, ensures orderly sequencing, and gives one entity full visibility of the
entire system.

---

## Core Architecture

```
                  ┌─────────────────────────────┐
  User goal ────► │       SUPERVISOR NODE        │
                  │  - reads task list           │
                  │  - selects next specialist   │
                  │  - tracks completed tasks    │
                  └──────────┬──────────────────┘
                    Command(goto=...)
           ┌────────────────┼───────────────────┐
           ▼                ▼                   ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  RESEARCHER  │  │   ANALYST    │  │    WRITER    │
   │  (sub-agent) │  │  (sub-agent) │  │  (sub-agent) │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                 │                  │
          └─────────────────┴──────────────────┘
                             │ all return to Supervisor
                             ▼
                  ┌─────────────────────────────┐
                  │       SUPERVISOR NODE        │
                  │  - collects result           │
                  │  - routes to next specialist │
                  │  - or routes to END          │
                  └─────────────────────────────┘
```

---

## State Design for Supervisor

```python
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class TaskPlan(BaseModel):
    tasks: list[str]            # ordered list of task descriptions
    current_index: int = 0      # which task is active
    assigned_to: Optional[str] = None  # which specialist is working

class SupervisorState(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    goal:        str
    task_plan:   Optional[dict]   # serialised TaskPlan
    results:     Annotated[list[str], operator.add]  # each specialist appends
    final_answer: Optional[str]
```

Key decisions:

- `results` uses `operator.add` so every specialist's output accumulates
- `task_plan` uses `dict` (not `TaskPlan`) because TypedDict values must be serialisable
- `messages` is shared across the full graph — all agents see the conversation history

---

## Supervisor Node

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SPECIALISTS = ["researcher", "analyst", "writer"]

def supervisor_node(state: SupervisorState) -> Command:
    """
    Supervisor decides who should act next (or whether to finish).
    Uses Command(goto=...) to route directly to a specific node.
    """
    # Build context from completed results
    results_context = ""
    if state.get("results"):
        results_context = "\n\nCompleted work so far:\n" + "\n---\n".join(state["results"])

    system = SystemMessage(
        "You are a supervisor managing three specialist agents: researcher, analyst, writer.\n"
        "Given the goal and the work completed so far, decide who should act next.\n"
        "When all work is complete, respond with 'FINISH'.\n"
        "Respond ONLY with the name of the next specialist ('researcher', 'analyst', 'writer') "
        "or 'FINISH'."
    )
    human = HumanMessage(
        f"Goal: {state['goal']}"
        + results_context
    )
    response = llm.invoke([system, human])
    decision = response.content.strip().lower()

    if decision == "finish" or decision not in SPECIALISTS:
        # All tasks complete — go to synthesiser
        return Command(goto="synthesiser")

    # Route to the chosen specialist
    return Command(
        goto=decision,
        update={"messages": [AIMessage(content=f"Supervisor assigned task to: {decision}")]},
    )
```

---

## Specialist Sub-Agent Nodes

Each specialist is a self-contained node. They receive the full state and return results
back to the Supervisor:

```python
def researcher_node(state: SupervisorState) -> Command:
    """Research specialist — gathers information on the goal."""
    system = SystemMessage(
        "You are a research specialist. Your job is to gather relevant facts and data. "
        "Be thorough, cite sources where possible, and stay focused on the goal."
    )
    human = HumanMessage(f"Research goal: {state['goal']}")
    response = llm.invoke([system, human])

    research_result = f"[RESEARCH]: {response.content}"

    # Always return to the supervisor after completing work
    return Command(
        goto="supervisor",
        update={
            "results":  [research_result],
            "messages": [AIMessage(content=research_result)],
        },
    )

def analyst_node(state: SupervisorState) -> Command:
    """Analysis specialist — interprets and structures findings."""
    prior_research = "\n".join(
        r for r in state.get("results", []) if r.startswith("[RESEARCH]")
    )
    system = SystemMessage(
        "You are an analysis specialist. Given the research findings, "
        "identify key patterns, themes, and strategic implications."
    )
    human = HumanMessage(
        f"Goal: {state['goal']}\n\nResearch findings:\n{prior_research}"
    )
    response = llm.invoke([system, human])

    analysis_result = f"[ANALYSIS]: {response.content}"
    return Command(
        goto="supervisor",
        update={
            "results":  [analysis_result],
            "messages": [AIMessage(content=analysis_result)],
        },
    )

def writer_node(state: SupervisorState) -> Command:
    """Writing specialist — produces the final deliverable."""
    all_findings = "\n\n".join(state.get("results", []))
    system = SystemMessage(
        "You are a professional writer. Given the research and analysis, "
        "produce a well-structured, engaging final deliverable."
    )
    human = HumanMessage(
        f"Goal: {state['goal']}\n\nFindings:\n{all_findings}"
    )
    response = llm.invoke([system, human])

    written_output = f"[FINAL DRAFT]: {response.content}"
    return Command(
        goto="supervisor",
        update={
            "results":  [written_output],
            "messages": [AIMessage(content=written_output)],
        },
    )

def synthesiser_node(state: SupervisorState) -> dict:
    """Combines all results into the final answer."""
    final_text = "\n\n".join(state.get("results", []))
    return {"final_answer": final_text}
```

---

## Graph Assembly

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(SupervisorState)

builder.add_node("supervisor",   supervisor_node)
builder.add_node("researcher",   researcher_node)
builder.add_node("analyst",      analyst_node)
builder.add_node("writer",       writer_node)
builder.add_node("synthesiser",  synthesiser_node)

builder.add_edge(START, "supervisor")
# Command(goto=...) handles all routing — no add_conditional_edges needed
# Each specialist returns Command(goto="supervisor") automatically
builder.add_edge("synthesiser", END)

graph = builder.compile(checkpointer=MemorySaver())

# Run the graph
config = {"configurable": {"thread_id": "supervisor-demo-001"}}
result = graph.invoke(
    {
        "goal": "Analyse the top 3 Python web frameworks for a startup in 2025.",
        "results": [],
        "final_answer": None,
    },
    config=config,
)
print(result["final_answer"])
```

---

## Preventing Runaway Delegation

A Supervisor without termination logic will keep assigning tasks indefinitely:

```python
MAX_SUPERVISOR_ROUNDS = 8

def supervisor_node(state: SupervisorState) -> Command:
    # Count how many times we've been through the supervisor
    supervisor_rounds = sum(
        1 for m in state["messages"]
        if isinstance(m, AIMessage) and m.content.startswith("Supervisor assigned")
    )

    if supervisor_rounds >= MAX_SUPERVISOR_ROUNDS:
        # Force termination
        return Command(
            goto="synthesiser",
            update={"messages": [AIMessage(content="[SUPERVISOR] Max rounds reached. Synthesising.")]},
        )

    # Normal routing logic continues...
```

---

## Common Pitfalls

| Pitfall                                           | Symptom                                                   | Fix                                                      |
| ------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------- |
| Specialist routes to another specialist directly  | Cycles; Supervisor loses visibility                       | All specialists must return `Command(goto="supervisor")` |
| No MAX_SUPERVISOR_ROUNDS guard                    | Infinite delegation loop                                  | Add a rounds counter to the Supervisor                   |
| `results` without `operator.add`                  | Only the last result is kept                              | Use `Annotated[list[str], operator.add]`                 |
| Supervisor uses `tool_calls` instead of `Command` | Supervisor becomes a tool-calling agent; no graph routing | Use `Command(goto=...)` for direct routing between nodes |
| Specialist state is isolated from shared state    | Specialist can't see prior work                           | All nodes share the same `SupervisorState`               |

---

## Mini Summary

- The Supervisor is the single routing authority; specialists never route to each other
- `Command(goto="specialist_name")` replaces conditional edges for dynamic routing
- All specialists return `Command(goto="supervisor")` after completing their work
- `results` uses `operator.add` so every specialist's output accumulates in state
- Always add a MAX_SUPERVISOR_ROUNDS guard to prevent infinite delegation

---

[← What Makes an Agent Deep](01-what-makes-an-agent-deep.md) | [Next → Swarm Handoff](03-swarm-handoff-pattern.md)
