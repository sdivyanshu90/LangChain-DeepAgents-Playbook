# 02 — Session Management

> **Previous:** [01 → Memory vs State](01-memory-vs-state.md) | **Next:** [03 → Selective Context Injection](03-selective-context-injection.md)

---

## Real-World Analogy

A hotel uses a room key card. Each card encodes a guest ID and an expiry.
The system doesn't re-ask who you are every time you use the lift.
It looks up your room assignment, your preferences, and your checkout date from the card.

Session management in AI applications works the same way:
a `session_id` is the key card — it maps a conversation to its stored history.

---

## What Session Management Solves

Without session management, every call to the LLM starts from zero:

```python
# WITHOUT sessions — stateless, no memory
llm = ChatOpenAI(model="gpt-4o-mini")
response1 = llm.invoke("My name is Alex.")
response2 = llm.invoke("What is my name?")
# response2: "I don't have information about your name."
```

With sessions, history is retrieved and injected automatically:

```python
# WITH sessions — stateful, memory persists
chain.invoke({"input": "My name is Alex."}, config={"configurable": {"session_id": "s1"}})
chain.invoke({"input": "What is my name?"},  config={"configurable": {"session_id": "s1"}})
# "Your name is Alex."
```

---

## `RunnableWithMessageHistory` — The Core Primitive

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# The session store: session_id → history object
# In production, replace with a persistent backend (see Topic 04 of Module 2.1)
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Factory function: given a session_id, return the history object for that session.
    Creates a new empty history if the session doesn't exist yet.
    This is the ONLY required contract for RunnableWithMessageHistory.
    """
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = RunnableWithMessageHistory(
    PROMPT | llm,
    get_session_history,
    input_messages_key="input",        # which key in invoke() dict is the user's message
    history_messages_key="history",    # which MessagesPlaceholder receives the history
)
```

---

## session_id Design

The `session_id` must uniquely identify a conversation.
It must be safe to use as a storage key.

```python
import secrets
import re

def new_session_id() -> str:
    """
    Generate a secure random session ID.
    Prefix with context so log entries are readable.
    """
    return f"session-{secrets.token_hex(12)}"
    # e.g., "session-4f3a2b8c1d9e7f6a"

def make_session_id(user_id: str, channel: str = "web") -> str:
    """
    Create a deterministic session ID from user + channel.
    Useful when one logical session per user per channel is needed.
    """
    raw = f"{user_id}-{channel}"
    # Sanitise: only alphanumeric and hyphens allowed as storage keys
    safe = re.sub(r"[^a-zA-Z0-9\-]", "-", raw)
    return safe

# Examples
print(new_session_id())                               # "session-4f3a2b8c1d9e7f6a"
print(make_session_id("u-8823", "web"))               # "u-8823-web"
print(make_session_id("alex@company.com", "slack"))   # "alex-company-com-slack"
```

**Security note:** A `session_id` used as a file path (for `FileChatMessageHistory`) must be sanitised.
Never pass raw user input directly as a session_id without sanitisation —
a session_id like `"../../etc/passwd"` would be a path traversal vulnerability.

---

## FileChatMessageHistory — Per-Session Persistent Files

```python
from langchain_community.chat_message_histories import FileChatMessageHistory
from pathlib import Path
import re, os

HISTORY_DIR = Path("./chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)

def sanitise_session_id(session_id: str) -> str:
    """Strip anything that could be used for path traversal."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
    return safe[:64]   # cap length to prevent excessively long filenames

def get_file_session_history(session_id: str) -> FileChatMessageHistory:
    """Return a file-backed history for the given session."""
    safe_id = sanitise_session_id(session_id)
    file_path = HISTORY_DIR / f"{safe_id}.json"
    return FileChatMessageHistory(str(file_path))

# Use exactly as before — just swap the factory function
file_chain = RunnableWithMessageHistory(
    PROMPT | llm,
    get_file_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Session persists across Python process restarts
file_chain.invoke(
    {"input": "Remember that my budget is $500."},
    config={"configurable": {"session_id": "user-42-web"}},
)
# Restart the process — history is loaded from disk automatically
file_chain.invoke(
    {"input": "What budget did I mention?"},
    config={"configurable": {"session_id": "user-42-web"}},
)
# "You mentioned your budget is $500."
```

---

## Managing Session Lifecycle

Sessions should not live forever. Implement TTL and cleanup:

```python
from datetime import datetime, timedelta
import json
from pathlib import Path

SESSION_TTL_DAYS = 30

def prune_old_sessions(history_dir: Path, ttl_days: int = SESSION_TTL_DAYS) -> int:
    """
    Delete session files not modified in the last `ttl_days` days.
    Returns the number of files pruned.
    Call this on a schedule (e.g., daily cron job).
    """
    cutoff = datetime.now() - timedelta(days=ttl_days)
    pruned = 0
    for session_file in history_dir.glob("*.json"):
        if datetime.fromtimestamp(session_file.stat().st_mtime) < cutoff:
            session_file.unlink()    # delete the file
            pruned += 1
    return pruned

# Example: run at startup or on a schedule
pruned = prune_old_sessions(HISTORY_DIR)
print(f"Pruned {pruned} expired sessions")
```

---

## Concurrent Sessions — What "Just Works"

Because each `session_id` maps to a separate history object,
concurrent sessions are safe by default when using in-memory or file-backed stores:

```python
# Concurrent users — each has their own history, completely isolated
session_a = "user-alice-web"
session_b = "user-bob-web"

# These can happen in any order or concurrently — no shared state
chain.invoke({"input": "My name is Alice."}, config={"configurable": {"session_id": session_a}})
chain.invoke({"input": "My name is Bob."},   config={"configurable": {"session_id": session_b}})

chain.invoke({"input": "What is my name?"},  config={"configurable": {"session_id": session_a}})
# "Your name is Alice."

chain.invoke({"input": "What is my name?"},  config={"configurable": {"session_id": session_b}})
# "Your name is Bob."
```

**Note:** For high-concurrency production workloads with Redis,
`RedisChatMessageHistory` serialises writes per session — no extra locking needed.
But if two requests with the **same** `session_id` arrive simultaneously,
the second write may overwrite the first.
Prevent this by ensuring each user's concurrent requests use different session IDs
or by queuing requests per session at the application layer.

---

## Inspecting the Session Store

During development, inspect the session store to debug history issues:

```python
def inspect_session(session_id: str) -> None:
    """Print a readable summary of a session's history."""
    history = get_session_history(session_id)
    messages = history.messages
    print(f"\nSession: {session_id}")
    print(f"Messages: {len(messages)}")
    for i, msg in enumerate(messages):
        role = type(msg).__name__.replace("Message", "").lower()
        print(f"  [{i+1}] {role}: {msg.content[:80]}")

inspect_session("user-42-web")
# Session: user-42-web
# Messages: 4
#   [1] human: Remember that my budget is $500.
#   [2] ai: I'll remember that. Your budget is $500.
#   [3] human: What budget did I mention?
#   [4] ai: You mentioned your budget is $500.
```

---

## Common Pitfalls

| Pitfall                                                  | What goes wrong                                               | Fix                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| Using unsanitised user input as session_id               | Path traversal attack via `../../` in session_id              | Always sanitise: `re.sub(r"[^a-zA-Z0-9_-]", "_", sid)`      |
| One global session for all users                         | All users share history; cross-contamination                  | Each user+channel combination must have a unique session_id |
| No session TTL                                           | History files accumulate indefinitely; disk fills up          | Implement `prune_old_sessions()` on a schedule              |
| Calling `get_session_history()` at startup for all users | Loads all history into memory unnecessarily                   | Only load history when a request for that session arrives   |
| Forgetting to pass `session_id` in config                | `RunnableWithMessageHistory` throws or uses a default session | Always pass `config={"configurable": {"session_id": sid}}`  |

---

## Mini Summary

- `RunnableWithMessageHistory` requires a factory function that maps `session_id → history object`.
- `session_id` must be sanitised before use as a file path or storage key.
- `FileChatMessageHistory` provides per-session file persistence that survives process restarts.
- Concurrent sessions with different session IDs are safely isolated by default.
- Implement TTL-based session cleanup — histories do not expire automatically.
- Always pass `config={"configurable": {"session_id": ...}}` on every invocation.
