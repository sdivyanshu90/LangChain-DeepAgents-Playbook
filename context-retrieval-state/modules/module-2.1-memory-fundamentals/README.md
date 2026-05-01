# Module 2.1 — Memory Fundamentals

> **Track:** Context, Retrieval & State | **Prerequisite:** Module 1.3 Structured Output

---

## The Central Insight

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Stateless LLM  +  Message List  =  Stateful Conversation App     │
│                                                                      │
│   ┌─────────┐      ┌──────────────────────────────────────────┐    │
│   │         │      │ [SystemMessage("You are helpful."),       │    │
│   │   LLM   │ ◄──  │  HumanMessage("Hi, I'm Alice."),         │    │
│   │         │      │  AIMessage("Hello Alice!"),               │    │
│   └────┬────┘      │  HumanMessage("What's my name?")]         │    │
│        │           └──────────────────────────────────────────┘    │
│        │                         ↑                                  │
│        ▼           This list IS the memory.                         │
│   AIMessage        The LLM is always stateless.                     │
│   ("Alice!")       Memory is an application-layer responsibility.   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Topics

| #   | File                                                                       | What you will learn                                          |
| --- | -------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 01  | [01-why-llms-are-stateless.md](01-why-llms-are-stateless.md)               | Why every LLM call starts fresh; the message list as memory  |
| 02  | [02-memory-strategies-compared.md](02-memory-strategies-compared.md)       | Full Buffer vs Window vs Summary — with token math           |
| 03  | [03-runnable-with-message-history.md](03-runnable-with-message-history.md) | `RunnableWithMessageHistory` deep dive                       |
| 04  | [04-storage-backends.md](04-storage-backends.md)                           | In-memory, File, and Redis backends with code                |
| 05  | [05-production-memory-patterns.md](05-production-memory-patterns.md)       | Multi-user isolation, summary compression, what NOT to store |

---

## Data Flow

```
User sends message
       │
       ▼
┌──────────────────┐
│  Load history    │  ← retrieve stored messages from backend (Redis / file / dict)
│  for session_id  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Assemble message list                                   │
│  [SystemMessage, ...history..., HumanMessage(new)]       │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────┐
│     LLM      │  (stateless — sees only this call's messages)
└────────┬─────┘
         │
         ▼
┌──────────────────┐
│  Save response   │  ← append HumanMessage + AIMessage to backend
│  to backend      │
└──────────────────┘
         │
         ▼
   Return AIMessage to caller
```

---

## Key Packages

```bash
pip install langchain-core langchain-openai langchain-community
# For Redis backend:
pip install redis
```

---

## How to Work Through This Module

1. Start with Topic 01 to internalise why statelessness matters.
2. Topic 02 gives you the vocabulary to choose a strategy.
3. Topics 03 and 04 are hands-on: run every code example.
4. Topic 05 covers production pitfalls — read it before building any real app.

A helpful colleague remembers what you already said, keeps the recent context in mind, and uses it to avoid making you repeat yourself. A stateless form forgets everything after each submission. LLM applications have the same problem: unless memory is designed explicitly, every interaction starts from zero.

## Why Memory Exists in LLM Applications

Memory is not there to make the model smarter. It exists to make the application context-aware.

Without memory, conversational systems degrade quickly:

- the user repeats information every turn
- prior decisions disappear
- follow-up questions lose meaning
- personalization becomes inconsistent

The key design question is not whether to add memory. It is what kind of state to keep, how much to keep, and how long to keep it.

## What “Memory” Actually Means

At this level, memory usually means one or more of the following:

- recent conversation history
- session-level variables such as user name, team, or workflow stage
- persisted summaries of prior interactions
- application state derived from previous turns

Why this distinction matters:

- not all memory should be replayed as raw chat history
- some context belongs in system instructions or user profile fields
- long histories eventually become expensive and noisy

## Core Memory Patterns

### 1. Full Conversation History

Best for:

- short sessions
- debugging early prototypes

Risk:

- token growth and irrelevant context accumulation

### 2. Windowed Memory

Best for:

- chat interfaces where only recent turns matter

Why it matters:

- keeps prompts smaller
- reduces distraction from stale context
- preserves enough continuity for follow-up questions

### 3. Persisted Session State

Best for:

- applications that need continuity across multiple requests or processes

Why it matters:

- a CLI, API, or web app can recover context by session identifier
- the memory layer becomes an application concern, not a prompt hack

## Internal Mechanics

Modern LangChain makes memory clearer by treating history as part of runnable execution instead of hiding it in opaque legacy abstractions.

The typical flow is:

1. Receive a session id.
2. Load message history for that session.
3. Inject that history into a prompt with a messages placeholder.
4. Run the model.
5. Store the new human and AI messages back into the session history.

That is a much better mental model than thinking of memory as magic.

## Example

See [examples/chat_session_state.py](examples/chat_session_state.py).

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

conversation = RunnableWithMessageHistory(
    prompt | model,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
```

That small wrapper is the important design step. The application now has a stable way to associate message history with a session.

## Best Practices

- store only the context that still helps the next step
- window or summarize long histories instead of replaying everything forever
- keep user profile data separate from conversational turns when possible
- use a stable session id so state recovery is predictable
- treat memory as application state, not as a side effect of prompting

## Common Pitfalls

- replaying the entire chat history by default
- confusing memory with retrieval from external knowledge
- stuffing profile data into every user turn instead of structured state
- assuming more context always improves quality

## Mini Summary

Memory is about continuity, not magic.

The application decides what to retain, how to retain it, and when that retained context should influence the next response.

## Optional Challenge

Extend the example so the session keeps only the last three exchanges, then compare how that changes prompt size and answer relevance.
