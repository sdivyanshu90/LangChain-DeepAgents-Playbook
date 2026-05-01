# 01 — Memory vs State

> **Previous:** [Module 2.4 README](README.md) | **Next:** [02 → Session Management](02-session-management.md)

---

## Real-World Analogy

A bank call centre agent has two windows open:

1. A call transcript — what the customer said in this call (conversation memory).
2. The account dashboard — customer name, plan, open tickets, account flags (application state).

The agent does NOT scroll through every previous call transcript to find out the customer's plan.
That information lives in the account dashboard — structured, queryable, reliable.

Your AI application needs the same separation.

---

## The Critical Distinction

```
CONVERSATION MEMORY
  ┌─────────────────────────────────────────────────────────┐
  │ Human: "I need to reset my password."                   │
  │ AI: "I've sent a reset link to your email."             │
  │ Human: "I don't see it."                                │
  │ AI: "Check your spam folder."                           │
  │ Human: "Still nothing."                                 │
  │ AI: "I can try SMS verification instead."               │
  └─────────────────────────────────────────────────────────┘
  → A LIST OF MESSAGES — the record of what was said
  → Naturally ordered, append-only
  → Grows over time

APPLICATION STATE
  ┌─────────────────────────────────────────────────────────┐
  │ user_id:           "u-8823"                             │
  │ user_name:         "Alex Chen"                          │
  │ user_role:         "engineer"                           │
  │ subscription:      "pro"                                │
  │ verified_email:    True                                 │
  │ email_on_file:     "alex@company.com"                   │
  │ sms_number:        "+1-555-0192"                        │
  │ failed_attempts:   2                                    │
  │ workflow:          "password_reset"                     │
  │ workflow_step:     3                                    │
  └─────────────────────────────────────────────────────────┘
  → STRUCTURED KEY-VALUE FACTS about the system's current knowledge
  → Updated by application logic (not by the LLM)
  → Random-access; queryable by field
```

---

## Why Mixing Them Is a Mistake

### Mistake 1 — Storing App State in Chat History

```python
# BAD: storing system facts as AI messages
history.add_ai_message("You are on the Pro plan.")
history.add_ai_message("Your account is verified.")

# These become part of the next prompt's context, but:
# - They grow the prompt each turn without adding new information
# - They can't be queried or validated without parsing message text
# - Older "AI" messages about state may be overridden by newer ones
#   but the old ones are still in the context, confusing the model
```

### Mistake 2 — Injecting Entire State Object Into Every Turn

```python
# BAD: dumping all state into every prompt regardless of step
state_str = json.dumps(user_state)
prompt = f"USER STATE: {state_str}\n\nHISTORY: {history}\n\nQuestion: {question}"

# Problems:
# - At step 3 of a password reset, why inject subscription tier?
# - At an FAQ step, why inject failed_attempts?
# - Every irrelevant field adds noise; model attends to noise
```

### Why This Matters

| Property         | Conversation Memory   | Application State                |
| ---------------- | --------------------- | -------------------------------- |
| Data type        | List of messages      | Structured dict / Pydantic model |
| Written by       | Human + AI turns      | Application logic only           |
| Growth pattern   | Append-only           | Updated in place                 |
| Query pattern    | Sequential read       | Field access by key              |
| Injection method | `MessagesPlaceholder` | Selective formatting             |
| Storage          | Chat history backend  | User/session store               |

---

## The Right Pattern

Keep memory and state in separate stores.
Inject them separately and selectively:

```python
from pydantic import BaseModel, Field
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# ── Application State (structured, validated) ──────────────────────────────
class PasswordResetState(BaseModel):
    user_id:          str
    user_name:        str
    user_role:        str
    verified_email:   bool
    email_on_file:    str
    failed_attempts:  int
    workflow_step:    int

# ── Chat History Store (append-only conversation record) ───────────────────
chat_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]

# ── Selective Context Injection (step-specific state fields) ───────────────
def format_state_for_step(state: PasswordResetState, step: int) -> str:
    """Return only the state fields needed for this workflow step."""
    if step <= 2:
        # Early steps: just identity
        return f"User: {state.user_name} ({state.user_role})"
    elif step == 3:
        # Email verification step: email + failure count
        return (
            f"User: {state.user_name}\n"
            f"Email on file: {state.email_on_file}\n"
            f"Verified: {state.verified_email}\n"
            f"Failed attempts: {state.failed_attempts}"
        )
    else:
        return f"User: {state.user_name} | Step: {step}"

# ── LLM and prompt ─────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a support assistant helping with password reset.\n\n"
        "CURRENT CONTEXT:\n{state_context}\n\n"
        "Respond based on the context above and the conversation history."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = RunnableWithMessageHistory(
    PROMPT | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ── Usage ──────────────────────────────────────────────────────────────────
user_state = PasswordResetState(
    user_id="u-8823",
    user_name="Alex Chen",
    user_role="engineer",
    verified_email=True,
    email_on_file="alex@company.com",
    failed_attempts=2,
    workflow_step=3,
)

response = chain.invoke(
    {
        "input": "I haven't received my reset email.",
        "state_context": format_state_for_step(user_state, user_state.workflow_step),
        # ↑ Only fields relevant to step 3 are in the prompt
        # ↑ Subscription tier, user_id, user_role are NOT included — not needed here
    },
    config={"configurable": {"session_id": "session-alex-001"}},
)
print(response.content)
```

---

## Recognising When Something Belongs in State vs History

Ask these questions:

```
Is it something that was said in this conversation?
  YES → conversation memory (append to history)
  NO  → continue

Is it a fact about the user, system, or current workflow?
  YES → application state (store in structured state)
  NO  → continue

Is it derived from the current AI response?
  YES → update application state after the call; do NOT add to history as an AI message
  NO  → continue

Is it a permanent preference or profile fact?
  YES → user profile store (separate from session state)
```

---

## Common Pitfalls

| Pitfall                                                            | What goes wrong                                                 | Fix                                                                         |
| ------------------------------------------------------------------ | --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Storing facts (like subscription tier) in chat history             | Facts repeat every turn; prompt grows unnecessarily             | Put facts in application state, inject via `state_context` field            |
| Injecting entire state object at every step                        | Irrelevant fields add noise; model's focus is diluted           | Use a step-specific `format_state_for_step()` function                      |
| Letting the LLM update application state by parsing its own output | State becomes unpredictable; model may hallucinate field values | Only update state from structured outputs (Pydantic) or deterministic logic |
| No separation: single `context` string for everything              | Can't control what changes or grows                             | Keep state and history as separate variables from the start                 |

---

## Mini Summary

- Conversation memory is a list of messages — what was said in this session.
- Application state is structured key-value facts — what the system knows about the user and workflow.
- Never store system facts as AI messages in chat history.
- Never dump the entire state object into every prompt.
- Use a step-specific selector to inject only the state fields needed for the current workflow step.
- Only update application state from structured outputs or deterministic application logic — never from freeform LLM text.
