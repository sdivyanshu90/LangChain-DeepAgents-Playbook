# 04 — Storage Backends

> **Previous:** [03 → RunnableWithMessageHistory](03-runnable-with-message-history.md) | **Next:** [05 → Production Memory Patterns](05-production-memory-patterns.md)

---

## Real-World Analogy

Three options for keeping a notebook:

1. **In your head** — fast to access, lost when you sleep (or the process restarts).
2. **On your desk** — persists overnight, but only accessible from your desk.
3. **In a shared cloud drive** — accessible from any device, by anyone with access.

In-memory, file, and Redis backends follow the same trade-off pattern.

---

## Why the Backend Choice Matters

The `get_session_history` function you pass to `RunnableWithMessageHistory`
determines where conversation history lives.
The wrong backend creates problems that are subtle and hard to debug in production:

```
Wrong backend → wrong failure mode

  InMemory in production → history lost on server restart; users complain
                            about "forgetting" conversations after deploys.

  File in multi-process  → process A writes alice.json; process B reads stale
                            or incomplete data; race conditions corrupt history.

  Redis misconfigured    → connection errors; sessions silently start empty;
                            no error is raised; model seems to have no memory.
```

Choose the backend that matches your deployment environment.

---

## Backend 1: InMemoryChatMessageHistory

**What it is:** A Python dict stored in process RAM.

**Lifecycle:** Data lives as long as the process lives. Restart = everything gone.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

# The store is a plain dict outside the history objects
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Create or retrieve in-memory history for this session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        # ^ Creates a new empty history for unknown session IDs
        # No persistence — if the process restarts, this session starts over
    return store[session_id]

# Usage:
history = get_session_history("user-123")
history.add_user_message("What is recursion?")
history.add_ai_message("Recursion is a function that calls itself...")

# Retrieve messages
for msg in history.messages:
    print(type(msg).__name__, ":", msg.content[:40])
# HumanMessage : What is recursion?
# AIMessage    : Recursion is a function that calls i...
```

**Trade-offs:**

| Dimension          | Value                                     |
| ------------------ | ----------------------------------------- |
| Setup complexity   | None — works out of the box               |
| Persistence        | None — lost on restart                    |
| Multi-process safe | No — each process has its own store dict  |
| Performance        | Fastest (in-process)                      |
| When to use        | Unit tests; prototypes; Jupyter notebooks |

---

## Backend 2: FileChatMessageHistory

**What it is:** JSON file on disk — one file per session.

**Lifecycle:** Persists across process restarts. Survives deploys.

```python
import os
from langchain_community.chat_message_histories import FileChatMessageHistory

HISTORY_DIR = "./session_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)   # ensure directory exists at startup

def get_file_session_history(session_id: str) -> FileChatMessageHistory:
    """
    Return a file-backed history for this session.
    The file is created automatically if it doesn't exist.
    session_id must be safe for use as a filename — sanitise it!
    """
    # SECURITY: sanitise session_id before using it as a filename
    # An unsanitised session_id could allow path traversal: "../../etc/passwd"
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe_id:
        raise ValueError(f"Invalid session_id: '{session_id}'")

    file_path = os.path.join(HISTORY_DIR, f"{safe_id}.json")
    return FileChatMessageHistory(file_path=file_path)

# Test it:
hist = get_file_session_history("user-alice")
hist.add_user_message("Hello")
hist.add_ai_message("Hi there!")

# Check what's on disk:
import json
with open("./session_histories/user-alice.json") as f:
    print(json.dumps(json.load(f), indent=2))
# [
#   {"type": "human", "data": {"content": "Hello", ...}},
#   {"type": "ai",    "data": {"content": "Hi there!", ...}}
# ]
```

### Clearing a Session's History

```python
# Remove all messages for a session (e.g., user requests "reset conversation")
hist = get_file_session_history("user-alice")
hist.clear()
# This deletes all messages but keeps the file (now empty [])

# To completely remove the session file:
import os
file_path = "./session_histories/user-alice.json"
if os.path.exists(file_path):
    os.remove(file_path)
```

**Trade-offs:**

| Dimension          | Value                                                             |
| ------------------ | ----------------------------------------------------------------- |
| Setup complexity   | Low — just a directory path                                       |
| Persistence        | Yes — survives process restart                                    |
| Multi-process safe | **No** — concurrent writes corrupt the file                       |
| Performance        | Slower than memory; adds file I/O per turn                        |
| When to use        | CLI tools; scripts with single-user; development with persistence |

---

## Backend 3: Redis (RedisChatMessageHistory)

**What it is:** Redis key-value store; each session stored as a Redis list.

**Lifecycle:** Persists until TTL expires or explicit deletion.

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7   # 7 days — expire inactive sessions

def get_redis_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    Return a Redis-backed history for this session.
    Creates the session automatically on first use.
    session_id should be a namespaced, stable identifier.
    """
    # Namespace the key to avoid collisions with other Redis data
    # e.g., "chat:user-123:session-456" instead of just "user-123"
    namespaced_id = f"chat:{session_id}"

    return RedisChatMessageHistory(
        session_id=namespaced_id,
        url=REDIS_URL,            # Redis connection URL from environment
        ttl=SESSION_TTL_SECONDS,  # auto-expire idle sessions after 7 days
    )

# Usage is identical to the other backends — same interface
hist = get_redis_session_history("user-alice:session-001")
hist.add_user_message("What is Redis?")
hist.add_ai_message("Redis is an in-memory data structure store...")

messages = hist.messages   # fetches from Redis; returns list of message objects
print(f"Stored {len(messages)} messages in Redis")
```

### Setting Up Redis Locally

```bash
# Option A: Docker (recommended for development)
docker run -d --name redis-dev -p 6379:6379 redis:7-alpine

# Option B: System Redis (Ubuntu/Debian)
sudo apt-get install redis-server
sudo systemctl start redis

# Test the connection
redis-cli ping   # should return: PONG
```

### Install the Python client

```bash
pip install redis langchain-community
```

### Why Namespace Session Keys

```
Without namespacing:
  Redis key: "alice"    ← conflicts with any other Redis usage of "alice"

With namespacing:
  Redis key: "chat:user:alice:session:001"
             └──┘ └──────────┘ └──────────┘
             app   user scope   session scope
```

**Trade-offs:**

| Dimension          | Value                                                         |
| ------------------ | ------------------------------------------------------------- |
| Setup complexity   | Medium — requires running Redis                               |
| Persistence        | Yes — survives restarts; TTL configurable                     |
| Multi-process safe | **Yes** — Redis serialises concurrent writes                  |
| Performance        | Fast (Redis is in-memory); adds network round-trip            |
| When to use        | Production APIs; multi-worker deployments; any multi-user app |

---

## Switching Backends with No Chain Changes

The power of `RunnableWithMessageHistory` is that you only change the backend function.
The chain itself is untouched:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

# Same chain, same prompt, same LLM
# Only the get_session_history function changes

# Development:
chain_dev = RunnableWithMessageHistory(
    chain, get_session_history,          # in-memory
    input_messages_key="input",
    history_messages_key="history",
)

# Staging (single process with persistence):
chain_staging = RunnableWithMessageHistory(
    chain, get_file_session_history,     # file-backed
    input_messages_key="input",
    history_messages_key="history",
)

# Production (multi-process):
chain_prod = RunnableWithMessageHistory(
    chain, get_redis_session_history,    # Redis
    input_messages_key="input",
    history_messages_key="history",
)
```

---

## Backend Selection Guide

```
Is this a unit test or a Jupyter notebook?
  → InMemoryChatMessageHistory

Is this a CLI tool or single-user script that should survive restarts?
  → FileChatMessageHistory

Is this a production web service with multiple workers or users?
  → RedisChatMessageHistory

Is this a serverless function (AWS Lambda, Cloud Functions)?
  → External storage is mandatory (Lambda has no persistent state between invocations)
  → Use Redis (ElastiCache) or DynamoDB
```

---

## Common Pitfalls

| Pitfall                            | What goes wrong                                             | Fix                                                 |
| ---------------------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| Unsanitised session_id as filename | Path traversal vulnerability in `FileChatMessageHistory`    | Always sanitise: keep only alphanumeric, `-`, `_`   |
| Hardcoded Redis URL                | Fails in different environments; credentials in source code | Use `os.getenv("REDIS_URL")`                        |
| No TTL on Redis sessions           | Redis fills up with abandoned sessions over months          | Set TTL; use `EXPIRE` for cleanup                   |
| File backend with multiple workers | Race conditions corrupt JSON files                          | Switch to Redis the moment you add a second process |
| No TTL on Redis sessions           | Abandoned sessions accumulate, filling Redis memory         | Always set `ttl=` in `RedisChatMessageHistory`      |

---

## Mini Summary

- Three backends: in-memory (dev), file (single-process persistence), Redis (production).
- All three share the same `BaseChatMessageHistory` interface — swap by changing one function.
- File backend: sanitise the `session_id` before using it as a filename.
- Redis backend: namespace keys, set a TTL, and read the URL from environment variables.
- Switch backends by replacing only the `get_session_history` function — the chain is unchanged.
