[← StateGraph Anatomy](01-stategraph-anatomy.md) | [Next → Edges and Routing](03-edges-and-routing.md)

---

# 02 — Nodes as Pure Functions

## Why Node Purity Matters

A LangGraph node is just a Python function. But the way LangGraph uses nodes creates
strong constraints that, if violated, produce subtle bugs:

- Nodes that mutate their input state cause race conditions in concurrent subgraphs
- Nodes that return the full State instead of a partial update cause reducer logic to be bypassed
- Nodes with side effects that depend on execution order become non-deterministic when retried

Understanding the contract of a node — what it must do and what it must not do — is the
foundation of reliable graphs.

---

## Real-World Analogy

A good assembly line worker picks up the partially-built product, adds their component,
and passes it on. They don't rebuild the entire product from scratch. They don't modify
the components already added by the previous worker. They don't store partially-built
products in their pocket for the next shift.

A LangGraph node is that assembly line worker: read the current state, add your
contribution, return only what you changed.

---

## The Node Contract

```
Node signature:  (state: State) -> dict

Input:  The FULL current state (read-only — don't mutate)
Output: A PARTIAL state update (only the keys this node changed)

The graph engine merges the partial update into the current state
using the reducer for each field.
```

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    retry_count: int

# ✅ CORRECT — returns only the keys this node updates
def classify_node(state: State) -> dict:
    last_message = state["messages"][-1]
    topic = _classify(last_message.content)
    return {"topic": topic}   # only "topic" changed; "messages" and "retry_count" untouched

# ❌ WRONG — returns the full State
def bad_node(state: State) -> State:
    new_state = dict(state)          # copies the state
    new_state["topic"] = "billing"
    return new_state                 # ← returns ALL keys; reducer for "messages" runs on existing messages → duplicates
```

---

## What the Graph Engine Does with Partial Updates

After a node returns `{"topic": "billing"}`, the graph engine:

1. For each key in the returned dict:
   - Looks up the reducer for that key in the State schema
   - Calls `reducer(current_value, new_value)` to produce the merged value
2. For keys NOT in the returned dict: current values are unchanged

```
Before node:    {"messages": [...], "topic": "",    "retry_count": 0}
Node returns:   {"topic": "billing"}
After merge:    {"messages": [...], "topic": "billing", "retry_count": 0}
                             ↑ reducer: "billing" replaces ""
                 ↑ unchanged   ↑ updated              ↑ unchanged
```

---

## Node Return Patterns

### Pattern 1 — Single Field Update

```python
def set_topic(state: State) -> dict:
    topic = determine_topic(state["messages"][-1].content)
    return {"topic": topic}
```

### Pattern 2 — Multiple Field Update

```python
def agent_node(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],         # add_messages reducer appends this
        "current_step": "tool_calling", # last-write-wins
    }
```

### Pattern 3 — Appending to a List (non-messages)

```python
class State(TypedDict):
    tool_results: Annotated[list[str], operator.add]

def tool_node(state: State) -> dict:
    result = run_tool(state)
    return {"tool_results": [result]}   # operator.add appends; wrap in list
```

### Pattern 4 — Incrementing a Counter

```python
class State(TypedDict):
    retry_count: Annotated[int, operator.add]

def retry_node(state: State) -> dict:
    return {"retry_count": 1}   # operator.add adds 1 to current count
```

### Pattern 5 — No Update (pass-through node)

```python
def logging_node(state: State) -> dict:
    print(f"[LOG] step={state['current_step']} retries={state['retry_count']}")
    return {}   # return empty dict — no state changes
```

---

## What Nodes Should NOT Do

### ❌ Mutating the Input State

```python
# WRONG:
def bad_node(state: State) -> dict:
    state["messages"].append(AIMessage("hello"))  # mutates the original list
    return {}  # the reducer never runs; the mutation is invisible to the graph engine
               # AND causes hard-to-trace bugs with concurrent subgraphs

# CORRECT:
def good_node(state: State) -> dict:
    return {"messages": [AIMessage("hello")]}  # reducer handles the append
```

### ❌ Reading Keys That Weren't Written Yet

```python
# WRONG (if "summary" is optional and may not be in state yet):
def use_summary(state: State) -> dict:
    text = state["summary"]   # KeyError if this node runs before summary is set

# CORRECT:
def use_summary(state: State) -> dict:
    text = state.get("summary", "")   # safe default
    return {"processed": process(text)}
```

### ❌ Performing Irreversible I/O Without a Guard

```python
# WRONG — sending an email inside a node that may be retried:
def notify_node(state: State) -> dict:
    send_email(state["user_email"], "Your report is ready")  # fires on every retry!
    return {}

# CORRECT — use the interrupt pattern (Module 3.4) for irreversible actions,
# or add an idempotency check:
def notify_node(state: State) -> dict:
    if state.get("email_sent"):
        return {}   # already sent; skip
    send_email(state["user_email"], "Your report is ready")
    return {"email_sent": True}
```

---

## Node Naming Best Practices

Node names become the node identifiers in routing and appear in LangSmith traces.
Use descriptive, verb-noun names:

```python
builder.add_node("classify_intent", classify_intent_node)   # ✅
builder.add_node("node1", fn1)                              # ❌ meaningless in traces
builder.add_node("call_llm", llm_node)                      # ✅
builder.add_node("llm", llm_node)                           # ❌ too generic
builder.add_node("execute_tools", tool_node)                # ✅
```

---

## Complete Example — Three Nodes with Different Update Patterns

```python
from typing import TypedDict, Annotated, Optional
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class SupportState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    category:     str
    confidence:   float
    retry_count:  Annotated[int, operator.add]
    final_answer: Optional[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Node 1: Classify the request category ─────────────────────────────────────
def classify_node(state: SupportState) -> dict:
    """Determine the category of the support request."""
    system = SystemMessage(
        "Classify this support request into exactly one of: billing, technical, account, general.\n"
        "Respond with only the category name."
    )
    response = llm.invoke([system, state["messages"][-1]])
    category = response.content.strip().lower()
    # Only update the fields this node owns:
    return {"category": category, "confidence": 0.9}

# ── Node 2: Generate a response ────────────────────────────────────────────────
def respond_node(state: SupportState) -> dict:
    """Generate a support response based on the classified category."""
    system = SystemMessage(
        f"You are a support agent specialising in {state['category']} issues. "
        "Provide a helpful, concise response."
    )
    response = llm.invoke([system, *state["messages"]])
    return {
        "messages": [response],       # add_messages appends the new AIMessage
        "final_answer": response.content,
    }

# ── Node 3: Log the interaction ────────────────────────────────────────────────
def log_node(state: SupportState) -> dict:
    """Log the completed interaction. Returns empty dict — no state changes."""
    print(f"[AUDIT] category={state['category']} tokens={len(state['messages'])}")
    return {}   # intentional empty return

# ── Graph assembly ─────────────────────────────────────────────────────────────
builder = StateGraph(SupportState)
builder.add_node("classify_intent", classify_node)
builder.add_node("generate_response", respond_node)
builder.add_node("log_interaction", log_node)

builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "generate_response")
builder.add_edge("generate_response", "log_interaction")
builder.add_edge("log_interaction", END)

graph = builder.compile()

# ── Run it ─────────────────────────────────────────────────────────────────────
initial_state = {
    "messages": [HumanMessage("My invoice shows the wrong amount. Please help.")],
    "category": "",
    "confidence": 0.0,
    "retry_count": 0,
    "final_answer": None,
}
result = graph.invoke(initial_state)
print(result["category"])       # "billing"
print(result["final_answer"])   # "I'd be happy to help with your billing issue..."
```

---

## Common Pitfalls

| Pitfall                                          | Symptom                                                     | Fix                                                   |
| ------------------------------------------------ | ----------------------------------------------------------- | ----------------------------------------------------- |
| Returning the full State dict                    | Reducer runs on already-merged values → duplicated messages | Return only the keys you changed                      |
| Mutating `state["messages"]` directly            | Reducer never applies; mutation invisible; race conditions  | Never mutate; always return `{"messages": [new_msg]}` |
| Accessing optional keys without `.get()`         | `KeyError` on first run before the key is set               | Use `state.get("key", default)`                       |
| Performing I/O that must be idempotent but isn't | Duplicate emails, database rows on retry                    | Add idempotency guards using state flags              |
| Very large logic in a single node                | Hard to test, trace, and debug                              | Split into focused single-responsibility nodes        |

---

## Mini Summary

- Node signature: `(state: State) -> dict` — read full state, return partial update
- The graph engine merges partial updates using the reducer defined in the `Annotated` annotation
- Never mutate the input `state`; never return the full State object
- Return `{}` from nodes that read state but don't update anything
- Use `state.get("key", default)` for optional fields that may not yet be set

---

[← StateGraph Anatomy](01-stategraph-anatomy.md) | [Next → Edges and Routing](03-edges-and-routing.md)
