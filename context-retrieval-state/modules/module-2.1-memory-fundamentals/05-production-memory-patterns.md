# 05 — Production Memory Patterns

> **Previous:** [04 → Storage Backends](04-storage-backends.md) | **Next:** [Module 2.2 → RAG End-to-End](../../module-2.2-rag-end-to-end/README.md)

---

## Real-World Analogy

A professional consultant keeps two types of records:

1. **Meeting notes** — what was said in each session.
2. **Client profile** — the client's industry, goals, budget, and constraints.

These are managed separately.
The profile informs every meeting but is not part of the meeting transcript.
Mixing them creates noise; separating them creates clarity.

LLM memory works the same way.

---

## Multi-User Session Isolation

The most critical production concern: **every user must have their own memory**.
Shared state between users is a privacy and correctness failure.

### The Isolation Pattern

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def get_isolated_session_history(
    user_id: str,
    session_id: str,
) -> RedisChatMessageHistory:
    """
    Create a Redis-backed history scoped to a specific user AND session.
    Two-level scoping: user_id prevents cross-user bleed;
    session_id allows one user to have multiple concurrent conversations.
    """
    # Compose a key that is unique per user AND per session
    redis_key = f"chat:{user_id}:{session_id}"
    # e.g. "chat:user:alice:session:2024-01-15:1"
    # Alice's sessions never overlap with Bob's even if session IDs collide

    return RedisChatMessageHistory(
        session_id=redis_key,
        url=REDIS_URL,
        ttl=60 * 60 * 24 * 7,  # 7 days
    )

# In a FastAPI endpoint:
# def chat_endpoint(request: ChatRequest, user: AuthenticatedUser):
#     history_getter = lambda session_id: get_isolated_session_history(
#         user_id=user.id,          # from authentication — never from request body
#         session_id=session_id,
#     )
#     return chain_with_history.invoke(
#         {"input": request.message},
#         config={"configurable": {"session_id": request.session_id}},
#     )
```

**Critical rule:** `user_id` must come from your authentication layer,
never from the request body. A user must not be able to claim another user's ID.

---

## Summary Compression for Long Sessions

Long sessions accumulate history that quickly exhausts token budgets.
Summarise periodically rather than waiting for the window to fill:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()
summariser_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SUMMARISE_AFTER_N_TURNS = 20   # trigger summarisation every 20 turns
KEEP_RECENT_N_TURNS     = 5    # always keep at least the last 5 turns in full

def maybe_compress_history(
    history: InMemoryChatMessageHistory,
    existing_summary: str = "",
) -> tuple[InMemoryChatMessageHistory, str]:
    """
    If the session is long enough, compress old turns into a summary.
    Returns: (updated history object, updated summary string)
    The history object is modified in-place: old turns removed, summary message prepended.
    """
    messages = history.messages

    # Only compress if we have enough turns to warrant it
    turn_count = len(messages) // 2   # each turn = 1 human + 1 AI message
    if turn_count < SUMMARISE_AFTER_N_TURNS:
        return history, existing_summary

    # Split: everything before the recent window, and the recent window
    keep_count = KEEP_RECENT_N_TURNS * 2   # message pairs → individual messages
    old_messages    = messages[:-keep_count]
    recent_messages = messages[-keep_count:]

    # Build the summarisation prompt
    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in old_messages
    )

    if existing_summary:
        summarisation_prompt = (
            f"Previous summary:\n{existing_summary}\n\n"
            f"New conversation to add:\n{conversation_text}"
        )
    else:
        summarisation_prompt = conversation_text

    summary_response = summariser_llm.invoke([
        SystemMessage(content=(
            "Produce a concise summary of the conversation. "
            "Preserve: names, key decisions, specific facts and numbers, open questions. "
            "Discard: greetings, filler, repetition."
        )),
        HumanMessage(content=summarisation_prompt),
    ])
    new_summary = summary_response.content

    # Rebuild history: summary message + recent messages only
    history.clear()
    history.add_message(
        SystemMessage(content=f"Summary of earlier conversation:\n{new_summary}")
    )
    for msg in recent_messages:
        history.add_message(msg)

    return history, new_summary
```

---

## Injecting User Profile State Separately from Chat History

Chat history records _what was said_.
User profile contains _what the system knows_ about the user.
These are different concerns and should be managed separately.

```python
from pydantic import BaseModel
from typing import Optional

class UserProfile(BaseModel):
    """Structured facts about the user, managed by the application (not the LLM)."""
    name: str
    role: str                   # e.g. "engineer", "manager", "student"
    team: Optional[str] = None
    skill_level: str = "beginner"   # "beginner", "intermediate", "expert"
    preferred_language: str = "English"

def build_system_message(profile: UserProfile) -> str:
    """
    Generate a context-rich system message from the user's profile.
    This is injected at the start of every conversation.
    It is NOT stored in the chat history — it's generated fresh each call.
    """
    return (
        f"You are a helpful assistant working with {profile.name}, "
        f"a {profile.skill_level} {profile.role}"
        + (f" on the {profile.team} team" if profile.team else "")
        + ".\n"
        f"Adjust explanation depth to the {profile.skill_level} level. "
        f"Respond in {profile.preferred_language}."
    )

# In your chain:
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

def get_prompt_with_profile(inputs: dict) -> list:
    """
    Build the full message list:
    1. System message from user profile (not from history)
    2. Stored conversation history
    3. Current user message
    """
    profile: UserProfile = inputs["profile"]
    history_messages = inputs.get("history", [])
    user_message = inputs["input"]

    return [
        SystemMessage(content=build_system_message(profile)),
        *history_messages,
        HumanMessage(content=user_message),
    ]

profile_aware_chain = RunnableLambda(get_prompt_with_profile) | llm
```

---

## What NOT to Put in Chat History

Chat history is not a general-purpose state store.
Putting the wrong things in history leads to inflated token usage and confused models.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     What belongs in chat history                    │
│                                                                     │
│  ✓  HumanMessage — exactly what the user typed                     │
│  ✓  AIMessage — exactly what the model responded                   │
│  ✓  SystemMessage from summariser — condensed summary of old turns │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                  What does NOT belong in chat history               │
│                                                                     │
│  ✗  User profile data (name, role, preferences)                    │
│     → Inject as a fresh system message each call instead            │
│                                                                     │
│  ✗  Application state (current workflow step, task status)         │
│     → Store in your database; inject as a system message if needed │
│                                                                     │
│  ✗  Retrieved documents from RAG                                   │
│     → Do not store in history; retrieve fresh each call            │
│                                                                     │
│  ✗  Raw tool call results (JSON payloads, API responses)           │
│     → Too verbose; summarise before adding to history              │
│                                                                     │
│  ✗  System prompts from your application                           │
│     → Always generate the system prompt dynamically; don't store it│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Full Production Pattern

Putting it all together:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# The base chain — no history management here
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_context}"),     # profile injected here dynamically
    MessagesPlaceholder("history"),      # stored history injected here
    ("human", "{input}"),
])

chain = prompt | llm

# Wrap with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_redis_session_history,           # production Redis backend
    input_messages_key="input",
    history_messages_key="history",
)

def chat(
    user_id: str,
    session_id: str,
    message: str,
    profile: UserProfile,
) -> str:
    """
    Production-ready chat function with:
    - Isolated user sessions
    - Profile context injected separately
    - History managed by RunnableWithMessageHistory
    """
    # Build system context from profile (not from history)
    system_context = build_system_message(profile)

    response = chain_with_history.invoke(
        {
            "input": message,
            "system_context": system_context,   # injected at call time
        },
        config={
            "configurable": {
                # Scope history to this specific user+session combination
                "session_id": f"{user_id}:{session_id}",
            }
        },
    )
    return response.content
```

---

## Common Pitfalls

| Pitfall                                                    | What goes wrong                                                | Fix                                                            |
| ---------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| Single global session for all users                        | All users share one conversation history; privacy disaster     | Always scope by `user_id + session_id`                         |
| Storing user profile in history                            | Profile repeated every turn; history bloat; stale profile data | Inject profile as system message at call time                  |
| Never compressing history                                  | Context overflow; cost explosions; degraded model quality      | Implement `maybe_compress_history` every 20-30 turns           |
| Trusting `session_id` from request body for user isolation | User can supply another user's `session_id`                    | Derive `user_id` from authentication token; never from request |
| Storing retrieved documents in history                     | History grows 5-10× faster; irrelevant docs re-shown to model  | Do not store RAG documents in chat history                     |

---

## Mini Summary

- Multi-user isolation requires both `user_id` (from auth) and `session_id` (from user) in the Redis key.
- Summary compression prevents context overflow on long sessions; trigger every 20-30 turns.
- User profile (name, role, preferences) is application state — inject as a system message, not stored in history.
- Chat history should contain only: `HumanMessage`, `AIMessage`, and summary `SystemMessage`.
- Retrieved documents, tool outputs, and system prompts do NOT belong in stored history.
