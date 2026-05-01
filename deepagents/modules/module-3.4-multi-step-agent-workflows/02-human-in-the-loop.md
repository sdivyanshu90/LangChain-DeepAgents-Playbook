[← ReAct Loop From Scratch](01-react-loop-from-scratch.md) | [Next → Send API Fan-Out](03-send-api-fan-out.md)

---

# 02 — Human-in-the-Loop

## Why Human-in-the-Loop Exists

Autonomous agents are powerful, but some actions are irreversible. Sending an email,
deleting a record, publishing content, or transferring funds cannot be undone by the agent.
If the model misunderstands the user's intent — which it will, eventually — the damage is real.

`interrupt()` gives you a surgical pause point: freeze the graph at an exact node,
surface the pending action to a human reviewer, and resume only after explicit approval.

---

## Real-World Analogy

A surgeon performing a complex operation has a scrub nurse who speaks up at critical
decision points: "We're about to ligate the wrong vessel — please confirm." The operation
doesn't continue until the surgeon confirms. The patient doesn't know this is happening;
the outcome is simply safer.

`interrupt()` is the scrub nurse. It pauses the agent at a defined critical point and
requires a human decision before proceeding.

---

## How `interrupt()` Works Mechanically

```
1. Agent node calls interrupt(value)
2. LangGraph raises NodeInterrupt exception (internally)
3. LangGraph catches it, saves the full State checkpoint
4. invoke() returns the current (paused) state to the caller
5. Caller inspects state.next to see which node is waiting
6. Human reviews the interrupt value
7. Caller calls graph.invoke(Command(resume=human_decision), config=config)
8. LangGraph reloads the checkpoint and resumes the paused node
9. The return value of interrupt() is now human_decision
10. Node continues from the line after interrupt()
```

---

## Basic Implementation

```python
# human_in_loop.py
from typing import TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class EmailState(TypedDict):
    messages:      Annotated[list[BaseMessage], add_messages]
    draft_email:   Optional[str]
    send_approved: bool
    final_status:  str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ── Node 1: Draft the email ────────────────────────────────────────────────────
def draft_email_node(state: EmailState) -> dict:
    """Generate an email draft based on the user's request."""
    system = SystemMessage(
        "You are an email writing assistant. "
        "Write a professional email based on the user's brief. "
        "Output ONLY the email body, no subject line, no commentary."
    )
    response = llm.invoke([system, *state["messages"]])
    return {
        "draft_email":   response.content,
        "send_approved": False,
    }

# ── Node 2: Human approval gate ───────────────────────────────────────────────
def approval_gate(state: EmailState) -> dict:
    """
    Pause for human review before sending the email.
    interrupt() freezes the graph here until Command(resume=...) is called.
    """
    # The value passed to interrupt() is surfaced to the human reviewer:
    decision = interrupt({
        "message":  "Please review this email draft before it is sent.",
        "draft":    state["draft_email"],
        "options":  ["approve", "revise: <new text>", "cancel"],
    })

    # Code below runs only AFTER the human resumes:
    if decision == "approve":
        return {"send_approved": True}
    elif decision == "cancel":
        return {"send_approved": False, "final_status": "cancelled_by_human"}
    else:
        # Human provided revised content (starts with "revise: ")
        if decision.startswith("revise: "):
            revised_draft = decision[len("revise: "):]
            return {"draft_email": revised_draft, "send_approved": True}
        # Unexpected response — treat as approval
        return {"send_approved": True}

# ── Node 3: Send the email ─────────────────────────────────────────────────────
def send_email_node(state: EmailState) -> dict:
    """Send the approved email. Only runs when send_approved is True."""
    if not state["send_approved"]:
        return {"final_status": "not_sent"}
    # Stub: replace with real email sending logic (SMTP, SendGrid, etc.)
    print(f"[EMAIL SENT]\n{state['draft_email']}")
    return {"final_status": "sent"}

# ── Routing ────────────────────────────────────────────────────────────────────
def after_approval(state: EmailState) -> str:
    if state.get("final_status") == "cancelled_by_human":
        return END
    if state["send_approved"]:
        return "send_email"
    return END

# ── Graph assembly ─────────────────────────────────────────────────────────────
builder = StateGraph(EmailState)
builder.add_node("draft_email",   draft_email_node)
builder.add_node("approval_gate", approval_gate)
builder.add_node("send_email",    send_email_node)

builder.add_edge(START, "draft_email")
builder.add_edge("draft_email", "approval_gate")
builder.add_conditional_edges("approval_gate", after_approval, {"send_email": "send_email", END: END})
builder.add_edge("send_email", END)

graph = builder.compile(checkpointer=MemorySaver())
```

---

## Running the Graph — Pause and Resume Flow

```python
config = {"configurable": {"thread_id": "email-session-001"}}

initial_state = {
    "messages":      [HumanMessage("Write a polite follow-up email to a vendor who hasn't replied in two weeks.")],
    "draft_email":   None,
    "send_approved": False,
    "final_status":  "",
}

# ── Step 1: Start the graph — it will pause at approval_gate ──────────────────
result = graph.invoke(initial_state, config=config)

# Check if the graph is paused:
state = graph.get_state(config)
print("Next nodes:", state.next)
# Output: ('approval_gate',)  ← graph is paused at this node

# Access the interrupt value:
interrupt_data = state.tasks[0].interrupts[0].value
print("Draft email for review:")
print(interrupt_data["draft"])

# ── Step 2: Human approves ────────────────────────────────────────────────────
final_result = graph.invoke(Command(resume="approve"), config=config)
print("Status:", final_result["final_status"])  # "sent"

# ── Alternative: Human requests revision ──────────────────────────────────────
# final_result = graph.invoke(
#     Command(resume="revise: Please add a specific deadline of May 15th."),
#     config=config
# )

# ── Alternative: Human cancels ────────────────────────────────────────────────
# final_result = graph.invoke(Command(resume="cancel"), config=config)
```

---

## `interrupt_before` vs `interrupt()` Inside Nodes

LangGraph provides two ways to add interrupts:

### Method 1: `interrupt()` Inside a Node (Flexible)

```python
# The node itself decides when and how to interrupt.
# Can pass rich context to the reviewer.
# Can branch based on the human's response.
def approval_gate(state):
    decision = interrupt({"draft": state["draft_email"], ...})
    if decision == "approve":
        return {"send_approved": True}
    ...
```

### Method 2: `interrupt_before` at Compile Time (Simple)

```python
# Interrupt before the node runs — the human is shown the current state.
# Less flexible: can't pass custom context to the reviewer.
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send_email"],   # pause before this node runs
)
# After pause: resume with Command(resume=None) to proceed, or don't resume to cancel.
```

**When to use which:**

- Use `interrupt()` inside the node when you need to pass a rich review payload or branch on the human's decision
- Use `interrupt_before` for simple "approve to proceed" gates where the current state is sufficient context

---

## Real Use Case: Email Draft Approval Before Sending

```
User:  "Send a meeting invite to the entire engineering team for next Friday at 2pm."

Agent workflow:
  1. draft_email_node  — generates the meeting invite text
  2. approval_gate     — interrupts; surfaces draft to sender for review
  ── PAUSED ──
  Human reviews: "Looks good, but change the time to 3pm."
  Human sends Command(resume="revise: Please change time to 3pm instead.")
  ── RESUMED ──
  approval_gate continues: updates draft_email, sets send_approved=True
  3. send_email_node  — sends the revised invite
```

---

## Common Pitfalls

| Pitfall                                                   | Symptom                                             | Fix                                                                                |
| --------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Calling `Command(resume=...)` without a pending interrupt | `ValueError: No pending interrupt`                  | Always check `state.next` is non-empty before resuming                             |
| `interrupt()` in a node that runs multiple times          | Pauses on every iteration                           | Add a `if not state.get("interrupt_done")` guard or use `interrupt_before` instead |
| Not persisting MemorySaver between script runs            | Checkpoint lost; can't resume after process restart | Use `SqliteSaver` or `PostgresSaver` for production                                |
| Human response not validated                              | Model receives malformed revision instruction       | Validate `decision` type and content before passing back                           |
| Using `interrupt_before` for complex decisions            | Can't pass context to reviewer; can't branch        | Use `interrupt()` inside the node for complex approval flows                       |

---

## Mini Summary

- `interrupt(value)` freezes the graph at the calling node, checkpoints state, and returns to the caller
- The caller inspects `graph.get_state(config).next` to detect the pause and `.tasks[0].interrupts[0].value` for context
- `Command(resume=value)` resumes the paused graph; `interrupt()` returns `value` in the node
- Use `interrupt()` inside the node for rich review flows; use `compile(interrupt_before=["node"])` for simple gates
- Always use a persistent checkpointer (`SqliteSaver`, `PostgresSaver`) when the human may respond hours later

---

[← ReAct Loop From Scratch](01-react-loop-from-scratch.md) | [Next → Send API Fan-Out](03-send-api-fan-out.md)
