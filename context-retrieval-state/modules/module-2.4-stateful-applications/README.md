# Module 2.4: Stateful Applications

> Part of the [Context, Retrieval & State](../../README.md) track.

---

## What This Module Covers

Memory tells you what was _said_.
State tells you what the system _knows_.

Most production AI applications need both — and keeping them cleanly separated
is the key skill this module teaches.

```
┌──────────────────────────────────────────────────────────────────────┐
│               CONVERSATION MEMORY  vs  APPLICATION STATE             │
├──────────────────────────┬───────────────────────────────────────────┤
│ Conversation Memory      │ Application State                         │
│ (what was said)          │ (what the system knows)                   │
├──────────────────────────┼───────────────────────────────────────────┤
│ Human: "reset password"  │ user_id: "u-8823"                        │
│ AI:    "check email"     │ user_role: "engineer"                    │
│ Human: "email not found" │ subscription: "pro"                      │
│ AI:    "try SSO login"   │ current_workflow: "password_reset"       │
│                          │ workflow_step: 3                         │
│                          │ verified_email: True                     │
│                          │ failed_attempts: 2                       │
├──────────────────────────┼───────────────────────────────────────────┤
│ Stored as: list of       │ Stored as: structured dict or Pydantic   │
│ BaseMessages             │ model; updated by application logic      │
│ Injected via:            │ Injected via: selective context          │
│ MessagesPlaceholder      │ injection function                       │
└──────────────────────────┴───────────────────────────────────────────┘
```

---

## Topics in This Module

| #                                       | Topic                       | Core concept                                                 |
| --------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| [01](01-memory-vs-state.md)             | Memory vs State             | The critical distinction and why mixing them causes problems |
| [02](02-session-management.md)          | Session Management          | Per-session stores, session_id design, concurrent sessions   |
| [03](03-selective-context-injection.md) | Selective Context Injection | Inject only what the current step needs                      |
| [04](04-multi-user-isolation.md)        | Multi-User Isolation        | Per-user scoping, why shared state is dangerous              |
| [05](05-context-pruning-strategies.md)  | Context Pruning Strategies  | Token budget management, selective retention, summarisation  |

---

## Data Flow: State-Aware Application

```
User Request
     │
     ▼
┌─────────────────────┐
│   Session Lookup    │  → Retrieve memory for session_id
│   (chat history)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  State Lookup       │  → Retrieve structured state for user_id
│  (application state)│    (role, subscription, workflow stage, ...)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Context Selector   │  → Pick only the state fields needed for
│                     │    this workflow step (not everything!)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                     PROMPT                              │
│  [system: role + task]                                  │
│  [state: user_role=engineer, workflow_step=3]           │
│  [history: last 5 turns of relevant conversation]       │
│  [human: current message]                               │
└─────────┬───────────────────────────────────────────────┘
          │
          ▼
       LLM Call
          │
          ▼
┌─────────────────────┐
│  State Updater      │  → Optionally update workflow_step, flags, etc.
└─────────┬───────────┘
          │
          ▼
      Response → User
```

---

## Key Packages

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    InMemoryChatMessageHistory,
    FileChatMessageHistory,
    RedisChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
```

---

## How to Use This Module

Work through topics 01 to 05 in order. Topic 01 establishes the conceptual distinction
(memory vs state) that all later topics build on.
If you skipped Module 2.1, read [Module 2.1 Topic 01](../module-2.1-memory-fundamentals/01-why-llms-are-stateless.md) first.

Consider a good customer portal. It knows who the user is, what plan they are on, which workflow they are in, and which details are safe to show at that moment.

That is more than conversation history. That is application state.

## Why Statefulness Matters

Many AI applications need more than chat memory. They need controlled access to session data, user preferences, workflow stage, and system-derived facts.

Statefulness matters because it lets the application answer with the right context, not just more context.

Examples of useful state:

- user role and team
- current task or workflow stage
- selected account, case, or project
- recent decisions or summaries
- safety or compliance constraints

## Memory vs State

This distinction matters.

Conversation memory is usually a record of what was said.

Application state is usually a structured record of what the system knows and should selectively inject.

Why this matters in production:

- state can be validated
- state can be scoped by permissions
- state can be partially injected instead of dumping everything into the prompt
- state makes behavior more predictable than replaying raw conversation alone

## Controlled Context Injection

The key idea is selective injection.

Do not push the entire state object into the prompt. Instead:

1. Decide which fields matter for the current step.
2. Format only those fields into the prompt.
3. Keep the rest in application state.

This reduces prompt noise and prevents accidental leakage of irrelevant or sensitive data.

## Internal Mechanics

A common pattern looks like this:

1. The application receives the request plus a state object.
2. A small preprocessing step selects the relevant state fields.
3. Those fields are injected into the prompt in a controlled format.
4. The model responds using the selected context.
5. The application optionally updates the state for the next step.

That is fundamentally different from treating the prompt as an all-purpose dumping ground.

## Example

See [examples/stateful_context_injection.py](examples/stateful_context_injection.py).

The example takes a small state object, extracts only the fields needed for the current response, and injects them into the prompt through a preprocessing runnable.

## Best Practices

- separate conversation history from structured application state
- inject only the fields needed for the current step
- keep state schemas explicit and stable
- use summaries for long-running sessions instead of replaying everything
- treat permissions and scoping as part of context design

## Common Pitfalls

- dumping the entire user profile into every prompt
- mixing transient chat turns with long-lived application facts
- forgetting to validate state fields before injection
- using raw memory when the real need is a structured state model

## Mini Summary

Stateful applications are not just chats with longer memory.

They are systems that manage structured context intentionally so the right information reaches the model at the right time.

## Optional Challenge

Take the example state object and add a permissions field. Then adjust the preprocessing step so restricted fields never reach the prompt.
