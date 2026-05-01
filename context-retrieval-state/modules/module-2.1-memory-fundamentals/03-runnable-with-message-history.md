# 03 — RunnableWithMessageHistory

> **Previous:** [02 → Memory Strategies Compared](02-memory-strategies-compared.md) | **Next:** [04 → Storage Backends](04-storage-backends.md)

---

## Real-World Analogy

Imagine a meeting facilitator who keeps a notebook per meeting room.
Before each meeting, they pull the room's notebook to brief attendees on prior decisions.
After the meeting, they add the new session's notes.

`RunnableWithMessageHistory` is that facilitator.
You hand it a chain; it handles retrieving and storing history automatically.
Your chain code stays clean — no history management mixed in.

---

## Why Not Manage History Manually?

You can manage the message list yourself (as shown in Topic 01).
But manual management leads to the same boilerplate in every function:

```python
# Without RunnableWithMessageHistory — the boilerplate problem:
def chat(session_id: str, user_message: str) -> str:
    history = load_history(session_id)        # you write this
    messages = build_messages(history, user_message)  # you write this
    response = chain.invoke(messages)
    save_turn(session_id, user_message, response)     # you write this
    return response.content
```

Three lines of plumbing for every conversational endpoint.
Multiply by N features and the boilerplate dominates the codebase.

`RunnableWithMessageHistory` wraps any chain and injects this lifecycle automatically.

---

## Architecture

```
RunnableWithMessageHistory
│
├── input_messages_key  ─────► which key in your dict is the user's new message
├── history_messages_key ────► which key the history is injected under (for prompts)
├── output_messages_key  ────► which key holds the AI response to be saved
│
└── get_session_history ─────► a callable that:
                               - accepts session_id: str
                               - returns a BaseChatMessageHistory object
                               - creates new history if session_id is unknown
```

On every `.invoke(input, config)`:

1. Calls `get_session_history(config["configurable"]["session_id"])`.
2. Loads existing messages from the returned history object.
3. Injects messages into the chain under `history_messages_key`.
4. Runs the chain.
5. Appends the new human turn + AI response to the history object.

---

## Basic Usage with InMemoryChatMessageHistory

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

# 1. Build the base chain (no history logic here)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),   # ← history injected here
    ("human", "{input}"),                           # ← current user message
])

chain = prompt | llm

# 2. Create the session store
#    A simple dict maps session_id → InMemoryChatMessageHistory
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return existing history for this session, or create a new one."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 3. Wrap the chain with history management
chain_with_history = RunnableWithMessageHistory(
    chain,                                 # the base chain to wrap
    get_session_history,                   # how to retrieve/create history
    input_messages_key="input",            # key in invoke() dict for user message
    history_messages_key="history",        # MessagesPlaceholder variable name
)

# 4. Invoke — always pass session_id in the config
config = {"configurable": {"session_id": "user-alice-session-1"}}

response_1 = chain_with_history.invoke(
    {"input": "My name is Alice and I'm learning Python."},
    config=config,
)
print(response_1.content)
# "Hello Alice! Python is a great choice..."

response_2 = chain_with_history.invoke(
    {"input": "What language did I say I'm learning?"},
    config=config,
)
print(response_2.content)
# "You mentioned you're learning Python."

# Different session_id = different conversation with no shared history
config_bob = {"configurable": {"session_id": "user-bob-session-1"}}
response_3 = chain_with_history.invoke(
    {"input": "What language am I learning?"},
    config=config_bob,
)
print(response_3.content)
# "I'm not sure — you haven't mentioned a programming language yet."
```

---

## The `MessagesPlaceholder` — Where History Is Injected

`MessagesPlaceholder` is the placeholder in a prompt template that receives a list of messages.

```python
from langchain_core.prompts import MessagesPlaceholder

# In your prompt template, MessagesPlaceholder marks the spot where the
# history message list will be inserted before the model call.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for a software company."),
    MessagesPlaceholder(variable_name="history"),  # ← entire history list inserted here
    ("human", "{input}"),
])

# At call time, the prompt becomes:
# SystemMessage("You are a helpful assistant...")
# HumanMessage("Hi, I'm Alice.")        ← from history
# AIMessage("Hello Alice!")             ← from history
# HumanMessage("What's my name?")       ← current input
```

The variable name in `MessagesPlaceholder(variable_name="history")`
**must match** `history_messages_key="history"` in `RunnableWithMessageHistory`.

---

## Inspecting the Session Store

```python
# After several conversation turns, inspect what is stored:
session = session_store["user-alice-session-1"]

print(f"Number of stored messages: {len(session.messages)}")
# Number of stored messages: 4 (2 turns = 2 human + 2 AI)

for msg in session.messages:
    role = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{role}]: {msg.content[:60]}")
# [User]: My name is Alice and I'm learning Python.
# [AI]:   Hello Alice! Python is a great choice...
# [User]: What language did I say I'm learning?
# [AI]:   You mentioned you're learning Python.
```

---

## FileChatMessageHistory — Persisting to Disk

`InMemoryChatMessageHistory` is lost when the process restarts.
For single-process persistence, use `FileChatMessageHistory`:

```python
from langchain_community.chat_message_histories import FileChatMessageHistory
import os

# Sessions are stored as JSON files in a directory
HISTORY_DIR = "./chat_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)   # create directory if not exists

def get_session_history_file(session_id: str) -> FileChatMessageHistory:
    """Return a file-backed history for this session_id."""
    # Each session gets its own file: ./chat_histories/alice-session-1.json
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    return FileChatMessageHistory(file_path=file_path)

chain_with_file_history = RunnableWithMessageHistory(
    chain,
    get_session_history_file,
    input_messages_key="input",
    history_messages_key="history",
)

# After this call, ./chat_histories/user-alice.json is created on disk
chain_with_file_history.invoke(
    {"input": "My name is Alice."},
    config={"configurable": {"session_id": "user-alice"}},
)

# Restart the process — history is still there because it's on disk
chain_with_file_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user-alice"}},
)
# "Your name is Alice." — loaded from the JSON file
```

---

## Understanding input_messages_key vs history_messages_key

These two parameters trip up many developers:

```python
RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",      # the new user message in your invoke() dict
    history_messages_key="history",  # where to inject stored history in the prompt
)

# When you call:
chain_with_history.invoke({"input": "Hello"}, config=...)
#                          ^^^^^^^^
#          This key must match input_messages_key="input"

# MessagesPlaceholder(variable_name="history") in your prompt
#                                    ^^^^^^^
#          This must match history_messages_key="history"
```

A mismatch between these causes a `KeyError` or empty history injection.

---

## Common Pitfalls

| Pitfall                                               | What goes wrong                                             | Fix                                                               |
| ----------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| Mismatched `variable_name` and `history_messages_key` | History not injected; model has no prior context            | Ensure both strings match exactly                                 |
| Missing `session_id` in config                        | `KeyError: 'session_id'`                                    | Always pass `config={"configurable": {"session_id": "..."}}`      |
| Using `InMemoryChatMessageHistory` in production      | History lost on process restart; no multi-process sharing   | Use `FileChatMessageHistory` or Redis backend                     |
| Not using `MessagesPlaceholder`                       | Prompt has no slot for history; it's silently ignored       | Add `MessagesPlaceholder(variable_name="history")` to your prompt |
| Wrapping a structured output chain                    | `RunnableWithMessageHistory` may conflict with tool calling | Test carefully; use `include_raw=True` to debug                   |

---

## Mini Summary

- `RunnableWithMessageHistory` wraps any chain and automates history load/inject/save.
- `get_session_history` maps a `session_id` to a `BaseChatMessageHistory` object.
- `MessagesPlaceholder` in the prompt is where the history list is injected.
- `input_messages_key` and `history_messages_key` must match your prompt and invoke dict.
- `InMemoryChatMessageHistory` is for development; `FileChatMessageHistory` for single-process persistence.
- Always pass `config={"configurable": {"session_id": "..."}}` on every `invoke()` call.
