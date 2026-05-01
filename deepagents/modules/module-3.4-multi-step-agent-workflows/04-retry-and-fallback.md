[← Send API Fan-Out](03-send-api-fan-out.md) | [Next → Termination Conditions](05-termination-conditions.md)

---

# 04 — Retry and Fallback

## Why Graph-Level Retry?

Tool-level retry (e.g., `@tool(retry_on_error=True)`) handles transient failures inside
a single tool call. But what if the entire _agent step_ fails? What if the model produces
malformed output, or a node raises an unexpected exception, or the response quality is
below a threshold?

Graph-level retry gives you control over the entire step: re-run the agent node,
re-run the tool node, or switch to a completely different strategy.

---

## Real-World Analogy

A business traveller's first flight is cancelled (transient failure). They try the same
route on the next departure (retry). If the next flight is also cancelled due to a weather
system (permanent failure), they switch to a different mode of transport: train or drive
(fallback).

Graph-level retry is the flight rebooking. Graph-level fallback is the train ticket.

---

## The Two Levels of Failure

```
Level 1: Tool-level failure
  A single tool call raises an exception.
  The agent receives a ToolMessage with an error.
  The agent may retry with corrected arguments automatically.
  → Handled in the tool node (Module 3.1/3.2)

Level 2: Graph-level failure
  The entire agent step fails (model error, quality check fails, node exception).
  A retry requires re-running a node, not just re-calling a tool.
  → Handled with retry_count in State + routing function
```

---

## Retry Counter in State

```python
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

MAX_RETRIES = 3  # module-level constant

class RetryState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    draft:        Optional[str]
    quality_score: float
    retry_count:  int     # last-write-wins (node sets it explicitly)
    error_detail: Optional[str]
    final_output: Optional[str]
```

Why `int` (not `Annotated[int, operator.add]`) for `retry_count`?

Because we want the routing function to compare `retry_count` to a threshold.
If we used `operator.add`, each invocation would accumulate across turns.
Here, `retry_count` tracks retries for the _current task_, so last-write-wins is correct.

---

## Routing Function That Checks Retries

```python
from langgraph.graph import END

def after_quality_check(state: RetryState) -> str:
    """
    Route based on quality score and retry count.
    - Good output → END
    - Poor output + retries left → retry writing
    - Poor output + no retries left → fallback
    """
    if state["quality_score"] >= 0.7:
        return "publish"

    if state["retry_count"] < MAX_RETRIES:
        return "write_draft"   # loop back to rewrite

    return "fallback_response"  # too many retries; escalate
```

---

## Complete Example — Retry + Fallback in a Writing Agent

```python
# retry_fallback.py
from typing import TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

MAX_RETRIES = 3

class WritingState(TypedDict):
    messages:      Annotated[list[BaseMessage], add_messages]
    draft:         Optional[str]
    critique:      Optional[str]
    quality_score: float
    retry_count:   int
    final_output:  Optional[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Node 1: Write draft ────────────────────────────────────────────────────────
def write_draft_node(state: WritingState) -> dict:
    """Generate or revise the draft. Uses critique from previous attempt if available."""
    critique_context = ""
    if state.get("critique"):
        critique_context = (
            f"\n\nPrevious attempt was insufficient. Critique:\n{state['critique']}\n"
            "Please address all critique points in your revised draft."
        )

    system = SystemMessage(
        "You are a professional content writer. Write clear, engaging, accurate content. "
        "Output only the content, no meta-commentary." + critique_context
    )
    response = llm.invoke([system, *state["messages"]])
    return {
        "draft":       response.content,
        "retry_count": state["retry_count"],   # unchanged here; incremented in quality_check
    }

# ── Node 2: Quality check ──────────────────────────────────────────────────────
def quality_check_node(state: WritingState) -> dict:
    """
    Assess the draft quality. Returns a score and critique.
    Quality score >= 0.7 is acceptable for publication.
    """
    import json

    system = SystemMessage(
        "You are an editorial quality assessor.\n"
        "Evaluate the given content on: clarity, accuracy, completeness, tone.\n"
        "Respond ONLY with JSON:\n"
        '{"score": <0.0-1.0>, "critique": "<specific improvement suggestions or \'acceptable\' if score>=0.7>"}'
    )
    user = HumanMessage(
        f"Original request: {state['messages'][0].content}\n\nDraft to assess:\n{state['draft']}"
    )
    response = judge_llm.invoke([system, user])

    try:
        parsed = json.loads(response.content)
        score   = float(parsed.get("score", 0.5))
        critique = str(parsed.get("critique", "No specific critique."))
    except (json.JSONDecodeError, ValueError):
        score, critique = 0.5, "Could not parse quality assessment."

    return {
        "quality_score": score,
        "critique":      critique,
        "retry_count":   state["retry_count"] + 1,  # increment after each quality check
    }

# ── Node 3: Publish ────────────────────────────────────────────────────────────
def publish_node(state: WritingState) -> dict:
    """Quality approved — finalise and 'publish'."""
    print(f"[PUBLISH] Quality score: {state['quality_score']:.2f}, retries: {state['retry_count']}")
    return {"final_output": state["draft"]}

# ── Node 4: Fallback ───────────────────────────────────────────────────────────
def fallback_node(state: WritingState) -> dict:
    """
    All retries exhausted. Return the best available draft with a notice,
    or escalate to a human reviewer.
    """
    print(f"[FALLBACK] All {MAX_RETRIES} retries exhausted. Score was {state['quality_score']:.2f}.")
    notice = (
        f"[NOTE: This content was produced after {MAX_RETRIES} revision attempts. "
        f"Final quality score: {state['quality_score']:.2f}. "
        "Manual review recommended.]\n\n"
    )
    return {"final_output": notice + (state["draft"] or "")}

# ── Routing ────────────────────────────────────────────────────────────────────
def route_after_quality(state: WritingState) -> str:
    """Route based on quality score and available retries."""
    if state["quality_score"] >= 0.7:
        return "publish"
    if state["retry_count"] < MAX_RETRIES:
        return "write_draft"      # loop: try again with critique
    return "fallback_response"    # give up gracefully

# ── Graph assembly ─────────────────────────────────────────────────────────────
builder = StateGraph(WritingState)
builder.add_node("write_draft",     write_draft_node)
builder.add_node("quality_check",   quality_check_node)
builder.add_node("publish",         publish_node)
builder.add_node("fallback_response", fallback_node)

builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "quality_check")
builder.add_conditional_edges(
    "quality_check",
    route_after_quality,
    {
        "write_draft":       "write_draft",       # retry loop
        "publish":           "publish",
        "fallback_response": "fallback_response",
    },
)
builder.add_edge("publish",           END)
builder.add_edge("fallback_response", END)

graph = builder.compile(checkpointer=MemorySaver())
```

---

## Distinguishing Transient vs Permanent Failures

```python
def smart_retry_routing(state: WritingState) -> str:
    """
    Distinguish between transient and permanent failures.

    Transient failures (worth retrying):
      - Quality score slightly below threshold (0.5-0.69)
      - Missing specific required section

    Permanent failures (use fallback immediately):
      - Quality score very low (<0.3) — model fundamentally misunderstood the task
      - Same low score after multiple retries — no improvement trend
    """
    score = state["quality_score"]
    retries = state["retry_count"]

    # Permanent failure: score too low to improve via retry
    if score < 0.3:
        return "fallback_response"

    # No improvement after 2+ retries — treat as permanent
    # (In production, you could check the score history stored in state)
    if retries >= 2 and score < 0.5:
        return "fallback_response"

    # Transient: marginal quality, retries remain
    if score < 0.7 and retries < MAX_RETRIES:
        return "write_draft"

    if score >= 0.7:
        return "publish"

    return "fallback_response"
```

---

## Common Pitfalls

| Pitfall                                                    | Symptom                                                      | Fix                                                                   |
| ---------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------- |
| `retry_count` using `operator.add`                         | Accumulates across multi-turn sessions; retries never happen | Use last-write-wins (plain `int`); set it explicitly in quality_check |
| No upper bound on retries                                  | Agent loops indefinitely on bad tasks                        | Always check `retry_count < MAX_RETRIES` in routing function          |
| Fallback node raises instead of returning output           | Process crashes after exhausting retries                     | Fallback must always produce a `final_output`; never raise            |
| Not including critique in the retry prompt                 | Model repeats same mistake on every retry                    | Pass `state["critique"]` to the write node on retry                   |
| Same routing function for transient and permanent failures | Permanent failures waste retries                             | Distinguish by score magnitude, not just pass/fail                    |

---

## Mini Summary

- Graph-level retry uses a `retry_count` field in State and a routing function that loops back below a threshold
- The routing function checks both quality (score >= threshold) AND retries (count < MAX_RETRIES) before looping
- A fallback node handles the "all retries exhausted" case — it must always produce output, never raise
- Pass the critique from the quality check node to the writing node on retry — otherwise the model repeats its mistakes
- Distinguish transient failures (retry) from permanent failures (immediately fallback) by score magnitude

---

[← Send API Fan-Out](03-send-api-fan-out.md) | [Next → Termination Conditions](05-termination-conditions.md)
