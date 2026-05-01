[← Edges and Routing](03-edges-and-routing.md) | [Next → Building a Complete Graph](05-building-a-complete-graph.md)

---

# 04 — MemorySaver and Checkpointing

## Why Checkpointing Exists

Without checkpointing, a LangGraph run is ephemeral. If the process crashes on step 6
of a 10-step agent, everything is lost and the user must start over. If the agent needs
a human to approve an action mid-run, there's nowhere to pause — either you block the
thread indefinitely or you lose context.

Checkpointing solves both problems by persisting the full State after _every node execution_.
This transforms graph runs from fragile, all-or-nothing transactions into resumable processes.

---

## Real-World Analogy

A video game save point. The game checkpoint is saved after every completed level.
If you lose connection or the console crashes, you don't restart from the beginning —
you restart from the last checkpoint. The `MemorySaver` is that save system.

`thread_id` is like a save-game slot — different players (sessions) have isolated saves.

---

## `MemorySaver` Setup

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# ── Build the graph as usual ───────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

# ── Compile WITH checkpointer ──────────────────────────────────────────────────
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Now every node execution automatically saves state to memory.
```

---

## `thread_id` — Session Isolation

Every `invoke()` or `stream()` call must pass a config dict with a `thread_id`.
This isolates state between different conversations or users:

```python
from langchain_core.messages import HumanMessage

# ── Session A ──────────────────────────────────────────────────────────────────
config_a = {"configurable": {"thread_id": "session-alice-001"}}

result_a = graph.invoke(
    {"messages": [HumanMessage("My order ORD-12345 is delayed.")]},
    config=config_a,
)

# Continue session A in the next turn — LangGraph loads Alice's state:
result_a2 = graph.invoke(
    {"messages": [HumanMessage("Can you check the tracking number?")]},
    config=config_a,
)

# ── Session B — completely isolated from A ─────────────────────────────────────
config_b = {"configurable": {"thread_id": "session-bob-002"}}
result_b = graph.invoke(
    {"messages": [HumanMessage("I want to cancel my subscription.")]},
    config=config_b,
)
```

**Key behaviour:** When you `invoke` with an existing `thread_id`, LangGraph loads the
last checkpointed state for that thread and _appends_ the new input to it. This is how
multi-turn conversations work without you manually managing message history.

---

## Inspecting Checkpointed State

```python
# Read the current state for a thread without running the graph:
state_snapshot = graph.get_state(config_a)
print(state_snapshot.values["messages"])     # All messages in this thread
print(state_snapshot.next)                   # Next nodes to execute ([] if done)
print(state_snapshot.created_at)             # Timestamp of last checkpoint

# See all checkpoints (full history) for a thread:
for checkpoint in graph.get_state_history(config_a):
    print(checkpoint.created_at, checkpoint.values.get("current_step", ""))
```

---

## `interrupt()` — Pausing for Human Review

`interrupt()` is LangGraph's human-in-the-loop primitive. When called inside a node,
it raises a `NodeInterrupt` exception. LangGraph catches this, saves the current
state (including the interrupt value), and returns to the caller.

The graph is paused at that node — not terminated — and can be resumed with `Command(resume=...)`.

```python
from langgraph.types import interrupt

def approval_gate_node(state: State) -> dict:
    """
    Pause for human approval before executing an irreversible action.
    The interrupt value is shown to the human reviewer.
    """
    draft_email = state["draft_email"]

    # This pauses the graph and surfaces the draft to the human:
    human_decision = interrupt({
        "message": "Please review this email before it is sent.",
        "draft": draft_email,
        "action_required": "approve or revise",
    })

    # Code here runs AFTER the human resumes the graph:
    if human_decision == "approve":
        return {"send_approved": True}
    else:
        # Human provided revised content:
        return {"draft_email": human_decision, "send_approved": False}
```

---

## Detecting That a Graph is Paused

When a graph hits an `interrupt()`, `invoke()` returns the current state with
`next` containing the interrupted node:

```python
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["send_email"])
# OR use interrupt() inside the node itself (more flexible)

result = graph.invoke(initial_state, config=config)

state = graph.get_state(config)
if state.next:
    print(f"Graph paused at: {state.next}")       # ('approval_gate_node',)
    print(f"Interrupt value: {state.tasks[0].interrupts[0].value}")  # the dict from interrupt()
```

---

## Resuming with `Command`

```python
from langgraph.types import Command

# Human approves the email:
graph.invoke(Command(resume="approve"), config=config)

# Human provides a revised draft:
graph.invoke(Command(resume="Please change the subject line to..."), config=config)
```

`Command(resume=value)` sets the return value of `interrupt()` in the paused node
and re-runs the graph from that point.

---

## Fault Tolerance — What Happens on Process Restart

`MemorySaver` stores state **in-process memory** — it does not survive a process restart.
For production systems where you need state to survive restarts, use a persistent backend:

```python
# PostgreSQL-backed checkpointer (requires langgraph-checkpoint-postgres):
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://user:pass@host/db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    # State survives process restarts; can be accessed from multiple processes

# SQLite-backed checkpointer (for local development):
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("./agent_state.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

---

## Complete Example — Multi-Turn Conversation with MemorySaver

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: ConversationState) -> dict:
    system = SystemMessage("You are a helpful assistant. Keep responses concise.")
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}

builder = StateGraph(ConversationState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "user-123"}}

# Turn 1:
result = graph.invoke({"messages": [HumanMessage("My name is Alex.")]}, config=config)
print(result["messages"][-1].content)   # "Nice to meet you, Alex! How can I help you?"

# Turn 2 — graph loads prior state automatically:
result = graph.invoke({"messages": [HumanMessage("What is my name?")]}, config=config)
print(result["messages"][-1].content)   # "Your name is Alex."
```

---

## Common Pitfalls

| Pitfall                                     | Symptom                                     | Fix                                                                            |
| ------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------ |
| Not passing `config` with `thread_id`       | Each `invoke()` starts a fresh state        | Always pass `config={"configurable": {"thread_id": "..."}}`                    |
| Using `MemorySaver` in production           | State lost on process restart               | Use `PostgresSaver` or `SqliteSaver` for persistence                           |
| Not checking `state.next` after interrupt   | Human-review step silently skipped          | Always check `graph.get_state(config).next` after interrupt workflows          |
| `Command(resume=...)` sent before interrupt | `ValueError` — no pending interrupt         | Only send `Command(resume)` when `state.next` is non-empty                     |
| Large state objects in MemorySaver          | High memory usage in long-running processes | Store large payloads (documents, images) externally; keep state IDs/references |

---

## Mini Summary

- `MemorySaver` checkpoints the full State after every node — enables multi-turn conversations and fault recovery
- `thread_id` in `configurable` isolates state per session; re-using a `thread_id` continues an existing conversation
- `interrupt(value)` pauses the graph mid-node and returns control to the caller; `Command(resume=value)` restores it
- For production persistence across process restarts, use `PostgresSaver` or `SqliteSaver`
- `graph.get_state(config)` lets you inspect or replay any checkpoint

---

[← Edges and Routing](03-edges-and-routing.md) | [Next → Building a Complete Graph](05-building-a-complete-graph.md)
