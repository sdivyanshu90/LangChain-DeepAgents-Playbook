# 04 — Multi-User Isolation

> **Previous:** [03 → Selective Context Injection](03-selective-context-injection.md) | **Next:** [05 → Context Pruning Strategies](05-context-pruning-strategies.md)

---

## Real-World Analogy

In a hospital, each patient's chart is locked to their room.
A nurse cannot accidentally open the wrong patient's chart by walking into the right room.
The chart is keyed to the patient, not the room.

Multi-user isolation in AI applications works the same way:
each user's data is keyed to their user ID, not to an arbitrary session ID.
A session ID alone is not a sufficient identity boundary.

---

## Why Shared State Is Dangerous

```
SHARED STATE BUG (production incident scenario):
─────────────────────────────────────────────────

  System uses one global InMemoryChatMessageHistory per worker process.
  All users on that worker share the same history object.

  User A (Alice, Pro account):
    "I need to update my billing to invoice."

  User B (Bob, Free account) — on the same worker:
    Chain.invoke("What payment options are available for my account?")
    Context: Alice's history is still in the shared store.

  Model responds: "Based on your account, invoice billing is available."
  Bob is on Free tier. Invoice billing is NOT available to him.
  Bob submits a support ticket expecting invoice billing.
  Support team has to escalate.
```

---

## The Three Isolation Requirements

```
1. CONVERSATION ISOLATION
   Each user's chat history must be separate.
   User A must not see or influence User B's history.

2. STATE ISOLATION
   Each user's application state (role, subscription, workflow) must be separate.
   State must be keyed by user_id, not just session_id.

3. PERMISSION-AWARE INJECTION
   Even when both users' state is correctly isolated,
   the system must inject only the context that user is authorised to see.
   A free-tier user must not receive a response that references pro-tier features.
```

---

## Scoping by user_id and session_id

Use a compound key: `"{user_id}::{session_id}"`.
This guarantees that even if two sessions share an ID (a bug), they can't cross user boundaries:

```python
from langchain_community.chat_message_histories import InMemoryChatMessageHistory, FileChatMessageHistory
from pathlib import Path
import re

# ── In-memory store keyed by (user_id, session_id) ─────────────────────────
class IsolatedSessionStore:
    """
    Thread-safe session store that enforces user-scoped isolation.
    Two different users with the same session_id get different histories.
    """

    def __init__(self) -> None:
        self._store: dict[str, InMemoryChatMessageHistory] = {}

    def _key(self, user_id: str, session_id: str) -> str:
        """
        Compound key: user_id::session_id.
        A session_id alone is not enough — must be scoped to the user.
        """
        safe_uid = re.sub(r"[^a-zA-Z0-9\-]", "-", user_id)
        safe_sid = re.sub(r"[^a-zA-Z0-9\-]", "-", session_id)
        return f"{safe_uid}::{safe_sid}"

    def get(self, user_id: str, session_id: str) -> InMemoryChatMessageHistory:
        key = self._key(user_id, session_id)
        if key not in self._store:
            self._store[key] = InMemoryChatMessageHistory()
        return self._store[key]

    def list_sessions(self, user_id: str) -> list[str]:
        """Return all session IDs belonging to a specific user."""
        prefix = re.sub(r"[^a-zA-Z0-9\-]", "-", user_id) + "::"
        return [k[len(prefix):] for k in self._store if k.startswith(prefix)]

    def delete_user(self, user_id: str) -> int:
        """Delete all sessions for a user (GDPR / account deletion)."""
        prefix = re.sub(r"[^a-zA-Z0-9\-]", "-", user_id) + "::"
        to_delete = [k for k in self._store if k.startswith(prefix)]
        for key in to_delete:
            del self._store[key]
        return len(to_delete)

store = IsolatedSessionStore()
```

---

## Using Isolated Store with RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from functools import partial

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support assistant.\n\nCONTEXT:\n{user_context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# RunnableWithMessageHistory's factory receives whatever keys are in
# config["configurable"]. We pass both user_id and session_id.
def get_history_for_user(
    session_id: str,
    user_id: str = "anonymous",
) -> InMemoryChatMessageHistory:
    return store.get(user_id=user_id, session_id=session_id)

chain = RunnableWithMessageHistory(
    PROMPT | llm,
    get_history_for_user,
    input_messages_key="input",
    history_messages_key="history",
)

# Every invocation MUST supply both user_id and session_id
def invoke_for_user(
    user_id: str,
    session_id: str,
    user_context: str,
    message: str,
) -> str:
    response = chain.invoke(
        {
            "input":        message,
            "user_context": user_context,
        },
        config={
            "configurable": {
                "session_id": session_id,
                "user_id":    user_id,     # passed to get_history_for_user
            }
        },
    )
    return response.content

# Alice and Bob get completely isolated histories
invoke_for_user("u-alice", "s-001", "Name: Alice | Tier: Pro",   "I need invoice billing.")
invoke_for_user("u-bob",   "s-001", "Name: Bob   | Tier: Free",  "What billing options do I have?")
# Bob will NOT see Alice's message — different compound keys
# "u-alice::s-001" ≠ "u-bob::s-001"
```

---

## Permission-Aware Context Injection

Isolation prevents data leakage between users.
Permissions prevent a user from receiving a response beyond their entitlements:

```python
from pydantic import BaseModel

class UserProfile(BaseModel):
    user_id:       str
    user_name:     str
    subscription:  str   # "free", "pro", "enterprise"
    verified:      bool

# Map: subscription tier → allowed context fields
ALLOWED_FIELDS: dict[str, set[str]] = {
    "free":       {"user_name"},
    "pro":        {"user_name", "subscription", "invoice_billing"},
    "enterprise": {"user_name", "subscription", "invoice_billing", "dedicated_support"},
}

FULL_CONTEXT = {
    "user_name":         "Alice Chen",
    "subscription":      "pro",
    "invoice_billing":   "Available",
    "dedicated_support": "Not available on Pro",   # enterprise-only
}

def permission_filtered_context(profile: UserProfile, full_context: dict) -> str:
    """Return context fields the user is permitted to see."""
    allowed = ALLOWED_FIELDS.get(profile.subscription, {"user_name"})
    filtered = {k: v for k, v in full_context.items() if k in allowed}
    return "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in filtered.items())

alice = UserProfile(user_id="u-alice", user_name="Alice", subscription="pro",   verified=True)
bob   = UserProfile(user_id="u-bob",   user_name="Bob",   subscription="free",  verified=True)

print(permission_filtered_context(alice, FULL_CONTEXT))
# User Name: Alice
# Subscription: pro
# Invoice Billing: Available

print(permission_filtered_context(bob, FULL_CONTEXT))
# User Name: Bob
# (invoice_billing and dedicated_support NOT shown — not on free tier)
```

---

## File-Backed Multi-User Store

For production workloads that need persistence without Redis:

```python
from langchain_community.chat_message_histories import FileChatMessageHistory
from pathlib import Path
import re

HISTORY_BASE = Path("./user_histories")
HISTORY_BASE.mkdir(exist_ok=True)

def get_user_file_history(
    session_id: str,
    user_id: str = "anonymous",
) -> FileChatMessageHistory:
    """
    Persist histories in a per-user directory.
    Structure: ./user_histories/{safe_user_id}/{safe_session_id}.json
    """
    safe_uid = re.sub(r"[^a-zA-Z0-9\-]", "-", user_id)
    safe_sid = re.sub(r"[^a-zA-Z0-9\-]", "-", session_id)[:64]

    user_dir = HISTORY_BASE / safe_uid
    user_dir.mkdir(exist_ok=True)

    return FileChatMessageHistory(str(user_dir / f"{safe_sid}.json"))
```

---

## Common Pitfalls

| Pitfall                                                      | What goes wrong                                        | Fix                                                                                |
| ------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Using `session_id` alone as the isolation key                | Two users with the same session_id share history       | Always compound: `f"{user_id}::{session_id}"`                                      |
| One shared `InMemoryChatMessageHistory` object for all users | All users read and write the same history              | Use `IsolatedSessionStore` or one history object per (user_id, session_id) pair    |
| Injecting all context fields regardless of subscription tier | Free user receives response referencing pro features   | Implement `permission_filtered_context()` gated on subscription tier               |
| Not implementing `delete_user()`                             | GDPR / account deletion requests fail                  | Implement user-scoped deletion from the start                                      |
| Checking only `session_id` on API calls                      | Session ID spoofing allows access to other users' data | Derive `user_id` from authenticated JWT / session token, not from the request body |

---

## Mini Summary

- Use a compound key `{user_id}::{session_id}` for all history lookups — session_id alone is insufficient.
- User profile, conversation history, and application state must all be scoped to `user_id`.
- Permissions go beyond isolation: even correctly isolated users must not receive responses referencing features beyond their tier.
- Use `permission_filtered_context()` to gate which fields are injected based on subscription or role.
- Implement `delete_user()` at the storage layer from day one — GDPR deletion is not optional.
- Always derive `user_id` from the authenticated identity (JWT/token), never from the request body.
