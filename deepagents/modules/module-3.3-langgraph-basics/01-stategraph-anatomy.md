[← Module Overview](README.md) | [Next → Nodes as Pure Functions](02-nodes-as-pure-functions.md)

---

# 01 — StateGraph Anatomy

## Why a Graph Instead of a Chain?

LCEL chains run left to right — one Runnable outputs to the next. This is perfect for
deterministic, stateless pipelines. But consider an agent that:

1. Calls a tool
2. Checks if the result is satisfactory
3. If not, calls a _different_ tool and loops back to step 2

An LCEL chain can't loop back. It can't share state across iterations without passing
the entire accumulated state as an argument through every step. And if it crashes on
step 4 of 8, you lose all work done in steps 1–3.

LangGraph solves all three problems:

- **Cycles:** edges can point backward, creating loops
- **Shared state:** every node reads from and writes to a single `State` object
- **Persistence:** state is checkpointed after every node execution

---

## Real-World Analogy

Think of a whiteboard in a war room. Every team member (node) walks up, reads the
current status (State), adds their contribution, then steps back. The next person
reads the updated whiteboard. If the building fire alarm goes off mid-meeting
(process crash), the whiteboard (checkpointed State) preserves everything when
everyone returns.

An LCEL chain is like passing sticky notes from person to person — works fine until
someone drops one.

---

## TypedDict State

LangGraph State is defined as a `TypedDict`. Every node receives the full state dict
and returns a partial update:

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # Annotated[type, reducer] specifies how incoming updates are merged
    messages: Annotated[list[BaseMessage], operator.add]
    topic: str                                            # last-write-wins
    retry_count: Annotated[int, lambda current, new: new] # also last-write-wins
    should_continue: bool
```

The `TypedDict` is not just documentation — LangGraph uses it to:

1. Validate state structure at compile time
2. Know which keys each node may update
3. Apply reducers when merging partial updates from multiple concurrent nodes

---

## Annotated Reducers — Why They Exist

Without reducers, two nodes running concurrently would clobber each other's updates.
The reducer defines the _merge strategy_ for each field.

### The Last-Write-Wins Problem

```
Node A writes: state["messages"] = [msg1, msg2, msg3]
Node B writes: state["messages"] = [msg4]

Without reducer: state["messages"] = [msg4]  ← Node A's work is lost ❌
With operator.add: state["messages"] = [msg1, msg2, msg3, msg4]  ✅
```

```python
from typing import Annotated
import operator

# operator.add: new value is appended to existing list
messages: Annotated[list, operator.add]

# lambda: custom merge logic
total_cost: Annotated[float, lambda old, new: old + new]   # accumulator

# No annotation: last-write-wins (fine for scalars)
current_step: str      # fine — only one node updates this at a time
```

---

## The `add_messages` Reducer

`add_messages` is LangChain's purpose-built reducer for message lists.
It does more than `operator.add`:

1. Appends new messages to the list
2. If a message with the same `id` already exists, **updates it** instead of duplicating it

```python
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Using add_messages instead of operator.add is best practice for message lists
```

Why `add_messages` over `operator.add`?

```python
# With operator.add: duplicate messages accumulate
# [HumanMessage("hello"), HumanMessage("hello")] ← both kept even if same id

# With add_messages: updates are idempotent
# Returning AIMessage(id="msg-1", content="v2") replaces AIMessage(id="msg-1", content="v1")
# This matters when nodes update existing messages (e.g., streaming)
```

---

## Building a StateGraph

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ── 1. Define State ────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ── 2. Define nodes ────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: State) -> dict:
    """Single-turn chat node: pass messages to LLM, return response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ── 3. Build the graph ─────────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("chat", chat_node)   # register the node

# ── 4. Connect the graph ───────────────────────────────────────────────────────
builder.add_edge(START, "chat")        # START is the entry sentinel
builder.add_edge("chat", END)          # END is the exit sentinel

# ── 5. Compile ─────────────────────────────────────────────────────────────────
graph = builder.compile()

# ── 6. Invoke ──────────────────────────────────────────────────────────────────
result = graph.invoke({"messages": [HumanMessage("What is LangGraph?")]})
print(result["messages"][-1].content)
```

---

## The Compiled Graph Object

`builder.compile()` returns a `CompiledStateGraph` — a Runnable you can invoke, stream, or
pass to other Runnables:

```python
graph = builder.compile()

# Invoke: returns final state
final_state = graph.invoke({"messages": [HumanMessage("Hello")]})

# Stream: yields state updates after each node
for state_update in graph.stream({"messages": [HumanMessage("Hello")]}, stream_mode="values"):
    print(state_update["messages"][-1])

# Async invoke:
import asyncio
final_state = asyncio.run(graph.ainvoke({"messages": [HumanMessage("Hello")]}))

# Visualise (requires graphviz or mermaid):
print(graph.get_graph().draw_mermaid())
```

---

## State Schema Validation at Compile Time

LangGraph validates your graph structure at `compile()` time, not at `invoke()` time.
Common compile-time errors:

```python
# Error: node "summarize" is referenced in an edge but never added
builder.add_edge("classify", "summarize")  # ← "summarize" not added as a node
graph = builder.compile()                  # raises ValueError: unknown node 'summarize'

# Error: no path from START to END
builder.add_node("classify", classify_node)
# Forgot add_edge(START, "classify") and add_edge("classify", END)
graph = builder.compile()  # raises ValueError or hangs on invoke
```

---

## Complete Annotated State — Production Pattern

```python
import operator
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ProductionState(TypedDict):
    # Message history — use add_messages for correct update semantics
    messages: Annotated[list[BaseMessage], add_messages]

    # Scalar fields — last-write-wins is fine
    user_id: str
    session_id: str
    topic: str
    current_step: str

    # Counter — accumulate across retries
    retry_count: Annotated[int, operator.add]

    # Cost tracking — accumulate token usage
    total_tokens: Annotated[int, operator.add]

    # Optional fields — not always present
    error_message: Optional[str]
    final_answer: Optional[str]

    # Boolean flags — last-write-wins
    should_escalate: bool
    is_complete: bool
```

---

## Common Pitfalls

| Pitfall                                     | Symptom                                                   | Fix                                                                   |
| ------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------- |
| Using `list` without `Annotated[list, ...]` | Node A's list update overwrites Node B's                  | Always annotate list fields with `add_messages` or `operator.add`     |
| Mutating `state` directly in a node         | Unpredictable state; reducer may apply on top of mutation | Return a new partial dict; never mutate `state`                       |
| Forgetting `START` and `END` edges          | Graph compiles but hangs or raises at invoke              | Always `add_edge(START, "first_node")` and route to `END`             |
| Very large State objects                    | Memory pressure at checkpoint; slow serialisation         | Keep State lean; store large objects (documents, images) by reference |
| TypedDict with Optional fields not set      | `KeyError` when node reads an unset Optional              | Provide defaults in initial state or use `state.get("key")`           |

---

## Mini Summary

- `TypedDict` State is the single shared whiteboard that all nodes read from and write to
- `Annotated[type, reducer]` defines how concurrent or sequential updates are merged — prevents last-write-wins data loss
- `add_messages` is the recommended reducer for message lists: it appends new messages and updates existing ones by ID
- `START` and `END` are built-in sentinels for graph entry and exit
- `builder.compile()` validates structure at compile time and returns a Runnable

---

[← Module Overview](README.md) | [Next → Nodes as Pure Functions](02-nodes-as-pure-functions.md)
