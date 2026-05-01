# 03 — Selective Context Injection

> **Previous:** [02 → Session Management](02-session-management.md) | **Next:** [04 → Multi-User Isolation](04-multi-user-isolation.md)

---

## Real-World Analogy

A doctor preparing for a consultation doesn't read the patient's entire medical record every time.
They read the section relevant to today's appointment:
a cardiologist visit pulls cardiac history; a dermatologist visit pulls skin conditions.

Selective context injection applies the same logic to your AI system:
inject the context relevant to the current step — not everything you have.

---

## Why Selective Injection Matters

```
Scenario: Support chatbot, 5 workflow steps.

Application state (all fields):
  user_id, user_name, user_email, user_role, subscription_tier,
  team_name, account_created, last_login, payment_method,
  open_tickets, ticket_priority, workflow_name, workflow_step,
  verified_identity, failed_attempts, preferred_language, timezone

Step 1: Greet the user and identify the issue
  Fields needed: user_name, preferred_language
  Fields NOT needed: open_tickets, ticket_priority, failed_attempts, payment_method

Step 3: Escalate a billing issue
  Fields needed: user_name, subscription_tier, payment_method, open_tickets
  Fields NOT needed: preferred_language, timezone, workflow_step

Injecting all 17 fields at every step:
  → ~500 extra tokens per call
  → Model attends to irrelevant fields
  → Answers reference fields that should not have been visible
  → Permission violation: payment_method visible at a non-billing step
```

---

## The Context Selector Pattern

Define the fields each workflow step needs, then enforce it:

```python
from pydantic import BaseModel, Field
from typing import Any

class UserState(BaseModel):
    """Full application state for a support user."""
    user_id:           str
    user_name:         str
    user_email:        str
    user_role:         str
    subscription:      str     # "free", "pro", "enterprise"
    team_name:         str
    payment_method:    str     # "card", "invoice"
    open_tickets:      int
    workflow:          str
    workflow_step:     int
    verified_identity: bool
    failed_attempts:   int
    preferred_language: str

# Map: workflow name + step → list of state fields to inject
CONTEXT_SCHEMA: dict[str, dict[int, list[str]]] = {
    "support": {
        1: ["user_name", "preferred_language"],
        2: ["user_name", "user_role", "subscription"],
        3: ["user_name", "subscription", "payment_method", "open_tickets"],
        4: ["user_name", "verified_identity", "failed_attempts"],
        5: ["user_name"],
    },
    "onboarding": {
        1: ["user_name", "preferred_language"],
        2: ["user_name", "user_role", "team_name"],
        3: ["user_name", "subscription"],
    },
}

def select_context(state: UserState) -> dict[str, Any]:
    """
    Return only the state fields needed for the current workflow step.
    Falls back to minimal context if step is not in schema.
    """
    steps_for_workflow = CONTEXT_SCHEMA.get(state.workflow, {})
    allowed_fields = steps_for_workflow.get(state.workflow_step, ["user_name"])
    # ↑ Default to just user_name if step not found — fail safe, not fail open

    return {
        field: getattr(state, field)
        for field in allowed_fields
    }

def format_context(selected: dict[str, Any]) -> str:
    """Format the selected state fields as a readable string for the prompt."""
    lines = [f"{key.replace('_', ' ').title()}: {value}" for key, value in selected.items()]
    return "\n".join(lines)
```

---

## Before and After: Prompt Comparison

```
BEFORE (dump entire state):
  ──────────────────────────────────────────────────────────────
  CONTEXT:
  User Id: u-8823
  User Name: Alex Chen
  User Email: alex@company.com
  User Role: engineer
  Subscription: pro
  Team Name: Platform Team
  Payment Method: invoice
  Open Tickets: 3
  Workflow: support
  Workflow Step: 1
  Verified Identity: True
  Failed Attempts: 0
  Preferred Language: English
  ──────────────────────────────────────────────────────────────

  [at step 1, 11 of these 13 fields are irrelevant noise]

AFTER (selective injection at step 1):
  ──────────────────────────────────────────────────────────────
  CONTEXT:
  User Name: Alex Chen
  Preferred Language: English
  ──────────────────────────────────────────────────────────────

  [exactly what is needed; nothing else]
```

---

## Integrating with RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful support assistant.\n\n"
        "RELEVANT CONTEXT FOR THIS STEP:\n{step_context}\n\n"
        # ↑ step_context is pre-computed by select_context + format_context
        # It changes per step; the prompt template is stable.
        "Answer based on the context and conversation history."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

chain = RunnableWithMessageHistory(
    PROMPT | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def run_with_state(
    user_state: UserState,
    user_message: str,
    session_id: str,
) -> str:
    """
    Run one turn with selective context injection.
    The caller is responsible for updating user_state.workflow_step as needed.
    """
    # 1. Select context fields for the current step
    selected = select_context(user_state)

    # 2. Format them as readable text
    formatted = format_context(selected)

    # 3. Invoke the chain
    response = chain.invoke(
        {
            "input":        user_message,
            "step_context": formatted,
        },
        config={"configurable": {"session_id": session_id}},
    )
    return response.content

# Usage
user_state = UserState(
    user_id="u-8823",
    user_name="Alex Chen",
    user_email="alex@company.com",
    user_role="engineer",
    subscription="pro",
    team_name="Platform Team",
    payment_method="invoice",
    open_tickets=3,
    workflow="support",
    workflow_step=1,
    verified_identity=True,
    failed_attempts=0,
    preferred_language="English",
)

# Step 1: Context = user_name + preferred_language only
reply = run_with_state(user_state, "Hello, I need help with my account.", "s-alex-001")
print(reply)

# Advance to step 3 (billing issue)
user_state.workflow_step = 3

# Step 3: Context = user_name + subscription + payment_method + open_tickets
reply = run_with_state(user_state, "I was charged twice this month.", "s-alex-001")
print(reply)
# Model sees subscription="pro" and payment_method="invoice" — relevant
# Model does NOT see user_email, team_name, failed_attempts — not injected
```

---

## User Profile vs Session History vs Task State

Three types of context — inject them differently:

```python
# USER PROFILE: facts that never change session-to-session
# ────────────────────────────────────────────────────────
# user_name, user_role, preferred_language, timezone
# → Inject from a profile store (not from session history)
# → Inject at every turn (small set; always relevant)

# SESSION HISTORY: the record of this conversation
# ────────────────────────────────────────────────────────
# The messages exchanged so far
# → Injected automatically via MessagesPlaceholder
# → Use window buffer or summary for long conversations (see Topic 05)

# TASK STATE: structured facts about the current workflow
# ────────────────────────────────────────────────────────
# workflow, workflow_step, open_tickets, verified_identity, failed_attempts
# → Inject selectively via select_context()
# → Only inject fields needed for the current step

def build_prompt_context(
    profile: dict,        # static user profile
    task_state: UserState,
    step: int,
) -> str:
    """Combine profile + step-specific task state into a single context string."""
    profile_lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in profile.items()]
    task_lines = [
        f"{k.replace('_', ' ').title()}: {v}"
        for k, v in select_context(task_state).items()
        if k not in profile   # avoid duplicating profile fields
    ]
    parts = []
    if profile_lines:
        parts.append("User profile:\n" + "\n".join(profile_lines))
    if task_lines:
        parts.append("Current context:\n" + "\n".join(task_lines))
    return "\n\n".join(parts)
```

---

## Common Pitfalls

| Pitfall                                                                         | What goes wrong                                       | Fix                                                                                    |
| ------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Hardcoding field lists inside the prompt template                               | Can't reuse template across workflows                 | Use a separate `CONTEXT_SCHEMA` dict; pass formatted string to `{step_context}`        |
| Forgetting to update `workflow_step` as conversation progresses                 | Wrong context injected; step 1 context used at step 3 | Increment `workflow_step` in the application layer after each turn or intent detection |
| Injecting sensitive fields (e.g., `payment_method`) before identity is verified | Model uses billing data before user is authenticated  | Make `verified_identity` a gate: skip payment fields until `verified_identity=True`    |
| Building the context selector once and never revisiting it                      | Schema drifts from actual application steps           | Review `CONTEXT_SCHEMA` whenever a new workflow step is added                          |

---

## Mini Summary

- Selective context injection means injecting only the state fields needed for the current workflow step.
- Define `CONTEXT_SCHEMA`: a mapping of `{workflow → {step → [field_names]}}`.
- User profile (permanent facts), session history (messages), and task state (workflow facts) are three separate categories — inject them separately.
- Format selected state as readable text before injecting into the prompt.
- Sensitive fields (payment details, identity flags) should only be injected at steps where they are needed and after identity verification.
