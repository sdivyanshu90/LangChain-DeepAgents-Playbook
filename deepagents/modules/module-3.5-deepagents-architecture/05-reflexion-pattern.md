[← Plan and Execute](04-plan-and-execute.md) | [Next Module → Observability](../../module-3.6-observability-and-debugging/README.md)

---

# 05 — Reflexion Pattern

## Why Self-Reflection Matters

A standard agent produces output and stops. It has no mechanism to ask: "Is this
actually good?" The result is that output quality is non-deterministic — sometimes
excellent, sometimes weak — and the agent has no way to improve within a single run.

The Reflexion pattern introduces an internal quality-control loop: a Reviewer node
evaluates the output against explicit criteria, and if quality is insufficient,
the Writer revises with the critique. The loop continues until quality passes or
revision limits are reached.

This brings consistent output quality — not by making the model smarter, but by
giving it the opportunity to review and improve its own work.

---

## Real-World Analogy

A novelist who always hands their manuscript to their editor before publication.
The editor provides specific feedback ("Chapter 3 pacing is too slow; the antagonist
motivation is unclear"). The novelist revises. The editor reviews again. After
at most three rounds of revision, the manuscript goes to print — even if it's not
perfect — because revisions have diminishing returns.

The `MAX_REVISION_CYCLES` guard is the "it goes to print at round 3" rule.

---

## The Reflexion Loop

```
                  ┌─────────────────────────────┐
  User prompt ──► │          WRITER              │
                  │  Produces initial draft      │
                  └──────────────┬───────────────┘
                                 │ draft
                                 ▼
                  ┌─────────────────────────────┐
                  │          REVIEWER            │
                  │  Scores on multiple criteria │
                  │  Provides specific critique  │
                  └──────────────┬───────────────┘
                                 │
                    ┌────────────┴───────────────┐
            score >= threshold         score < threshold
            OR max revisions reached   AND revisions remain
                    │                            │
                    ▼                            ▼
               ─────────                  ┌────────────┐
               ACCEPTED                   │   WRITER   │ ← receives critique
               ─────────                  └────────────┘
                    │                       (loop back)
                    ▼
               FINAL OUTPUT
```

---

## State Design — Revision History

```python
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

MAX_REVISION_CYCLES = 3   # how many times the writer can be sent back

class ReviewScore(BaseModel):
    """Structured review result from the Reviewer node."""
    clarity:       float   # 0.0 to 1.0
    accuracy:      float   # 0.0 to 1.0
    completeness:  float   # 0.0 to 1.0
    tone:          float   # 0.0 to 1.0
    critique:      str     # specific improvement guidance
    overall_score: float   # weighted average

    @classmethod
    def from_llm_output(cls, raw: str) -> "ReviewScore":
        import json
        parsed = json.loads(raw)
        overall = (
            parsed["clarity"] * 0.25
            + parsed["accuracy"] * 0.35
            + parsed["completeness"] * 0.25
            + parsed["tone"] * 0.15
        )
        return cls(**parsed, overall_score=overall)

class ReflexionState(TypedDict):
    messages:         Annotated[list[BaseMessage], add_messages]
    prompt:           str
    current_draft:    Optional[str]
    revision_history: Annotated[list[dict], operator.add]  # list of ReviewScore dicts
    revision_count:   int
    final_output:     Optional[str]
```

Why `revision_history` with `operator.add`?
Each revision cycle appends a review score — this gives LangSmith a full trace of
quality improvement over time, and lets you debug when and why revisions were requested.

---

## Writer Node

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)  # slightly creative for writing

def writer_node(state: ReflexionState) -> dict:
    """
    First call: write from scratch.
    Subsequent calls: revise based on critique from last review.
    """
    critique_section = ""
    if state["revision_history"]:
        last_review = state["revision_history"][-1]
        critique_section = (
            f"\n\n## Revision Instructions\n"
            f"Your previous draft scored {last_review['overall_score']:.2f}/1.0.\n"
            f"The reviewer said:\n{last_review['critique']}\n\n"
            f"Address every point raised. Do not repeat the same weaknesses."
        )

    system = SystemMessage(
        "You are a professional content writer. Produce clear, accurate, engaging content.\n"
        "Write only the content — no meta-commentary, headers like 'Draft:', etc."
        + critique_section
    )
    human = HumanMessage(f"Write content for this prompt:\n{state['prompt']}")

    response = llm.invoke([system, human])
    return {
        "current_draft": response.content,
        "messages":      [response],
    }
```

---

## Reviewer Node

```python
reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def reviewer_node(state: ReflexionState) -> dict:
    """
    Evaluates the current draft on four dimensions and produces a critique.
    Returns structured ReviewScore persisted in revision_history.
    """
    import json

    system = SystemMessage(
        "You are an expert editorial reviewer.\n"
        "Evaluate the draft on four dimensions (each 0.0-1.0):\n"
        "  - clarity:      Is the writing clear and easy to understand?\n"
        "  - accuracy:     Is the content factually accurate and precise?\n"
        "  - completeness: Does it fully address the prompt?\n"
        "  - tone:         Is the tone appropriate for the context?\n\n"
        "Then write a specific 'critique' — concrete suggestions for improvement.\n"
        "If everything is good, write 'No major issues' as critique.\n\n"
        "Respond ONLY with JSON:\n"
        '{"clarity": 0.0, "accuracy": 0.0, "completeness": 0.0, '
        '"tone": 0.0, "critique": "..."}'
    )
    human = HumanMessage(
        f"Original prompt: {state['prompt']}\n\nDraft to review:\n{state['current_draft']}"
    )

    response = reviewer_llm.invoke([system, human])

    try:
        review = ReviewScore.from_llm_output(response.content)
    except Exception:
        # Fallback score if JSON parsing fails
        review = ReviewScore(
            clarity=0.6, accuracy=0.6, completeness=0.6, tone=0.6,
            critique="Review parsing failed. Attempting revision.",
            overall_score=0.6,
        )

    return {
        "revision_history": [review.model_dump()],
        "revision_count":   state["revision_count"] + 1,
        "messages":         [AIMessage(content=f"Review score: {review.overall_score:.2f}")],
    }
```

---

## Routing After Review

```python
from langgraph.graph import END

QUALITY_THRESHOLD = 0.80

def route_after_review(state: ReflexionState) -> str:
    """
    Accept output if quality is sufficient or max revisions reached.
    Otherwise send back to writer with the critique.
    """
    if not state["revision_history"]:
        return "writer"  # no review yet — shouldn't happen, but safe fallback

    last_review = state["revision_history"][-1]
    score = last_review["overall_score"]

    # Accepted: quality good enough
    if score >= QUALITY_THRESHOLD:
        return "accept"

    # Accepted with caveat: too many revisions
    if state["revision_count"] >= MAX_REVISION_CYCLES:
        return "accept"   # publish with a quality notice (see accept node)

    # Needs revision
    return "writer"
```

---

## Accept Node

```python
def accept_node(state: ReflexionState) -> dict:
    """
    Finalises the output. Adds a quality notice if max revisions were reached.
    """
    last_review = state["revision_history"][-1] if state["revision_history"] else {}
    score = last_review.get("overall_score", 1.0)

    if state["revision_count"] >= MAX_REVISION_CYCLES and score < QUALITY_THRESHOLD:
        notice = (
            f"\n\n---\n*Note: This content was produced after {MAX_REVISION_CYCLES} "
            f"revision cycles. Final quality score: {score:.2f}. "
            "Manual review is recommended.*"
        )
        output = (state["current_draft"] or "") + notice
    else:
        output = state["current_draft"] or ""

    return {"final_output": output}
```

---

## Complete Graph Assembly

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(ReflexionState)
builder.add_node("writer",   writer_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("accept",   accept_node)

builder.add_edge(START, "writer")          # start with writing
builder.add_edge("writer", "reviewer")    # always review after writing

builder.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "writer": "writer",   # loop: revise
        "accept": "accept",   # done
    },
)
builder.add_edge("accept", END)

graph = builder.compile(checkpointer=MemorySaver())

# Run
config = {"configurable": {"thread_id": "reflexion-demo-001"}}
result = graph.invoke(
    {
        "prompt":           "Explain the CAP theorem to a software engineering intern.",
        "revision_history": [],
        "revision_count":   0,
        "final_output":     None,
    },
    config=config,
)

print("Final output:\n", result["final_output"])
print("\nRevision history:")
for i, rev in enumerate(result["revision_history"], 1):
    print(f"  Cycle {i}: score={rev['overall_score']:.2f} — {rev['critique'][:80]}")
```

---

## Inspecting Revision History in LangSmith

Because `revision_history` accumulates with `operator.add`, every review cycle is
visible in the LangSmith trace. You can see:

- Was the score improving across cycles?
- Did the critique identify the same issues every time? (indicates a model limitation)
- How many revision cycles did a typical run require?

```python
# Programmatic cost analysis across revision cycles
def analyse_revision_efficiency(state: ReflexionState) -> dict:
    history = state["revision_history"]
    if len(history) < 2:
        return {"improved": False, "delta": 0.0}

    first_score = history[0]["overall_score"]
    last_score  = history[-1]["overall_score"]
    return {
        "cycles":      len(history),
        "start_score": first_score,
        "end_score":   last_score,
        "improved":    last_score > first_score,
        "delta":       last_score - first_score,
    }
```

---

## Common Pitfalls

| Pitfall                                | Symptom                                            | Fix                                                                      |
| -------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------ |
| No `MAX_REVISION_CYCLES`               | Infinite revision loop on hard tasks               | Always check `revision_count >= MAX_REVISION_CYCLES` in routing          |
| Reviewer never scores < threshold      | Writer never gets to revise; Reflexion is a no-op  | Lower `QUALITY_THRESHOLD` or test the reviewer prompt in isolation       |
| Critique not passed to Writer          | Writer produces identical output on every revision | Pass `state["revision_history"][-1]["critique"]` to Writer system prompt |
| `revision_count` using `operator.add`  | Count double-increments across turns               | Use last-write-wins: `"revision_count": state["revision_count"] + 1`     |
| Accept node raises when quality is low | Process crashes at the worst time                  | Accept node must always produce `final_output`; never raise              |

---

## Mini Summary

- Reflexion = generate → critique → revise loop; the Writer revises until the Reviewer is satisfied
- `ReviewScore` on multiple dimensions (clarity, accuracy, completeness, tone) produces better critiques than a single score
- `revision_history` with `operator.add` creates an auditable trail of quality improvement in LangSmith
- `MAX_REVISION_CYCLES` is mandatory; accept the best-available draft when it's reached
- The critique must be explicitly passed to the Writer on every revision — otherwise it repeats the same mistakes

---

[← Plan and Execute](04-plan-and-execute.md) | [Next Module → Observability](../../module-3.6-observability-and-debugging/README.md)
