# 01 — Why LLMs Are Stateless

> **Previous:** [README → Module Index](README.md) | **Next:** [02 → Memory Strategies Compared](02-memory-strategies-compared.md)

---

## Real-World Analogy

Every time you walk up to a bank teller with amnesia,
you hand them a complete folder: your ID, your account number, your request.
They process it perfectly. You come back five minutes later.
They have no idea who you are. You hand them the folder again.

This is exactly how an LLM API works.
The model has no persistent memory between API calls.
Every request must contain all the context the model needs.

---

## The Fundamental Design Decision

LLMs are stateless by design — not by accident.
Understanding why helps you design memory correctly.

```
Why stateless?

  ┌────────────────────────────────────────────────────────────┐
  │  Reason 1: Scalability                                     │
  │  Any server in a cluster can handle any request.           │
  │  No "sticky sessions" needed. Horizontal scaling is free.  │
  ├────────────────────────────────────────────────────────────┤
  │  Reason 2: Reproducibility                                 │
  │  The same input always produces the same output            │
  │  (at temperature=0). State stored server-side would        │
  │  make debugging nearly impossible.                         │
  ├────────────────────────────────────────────────────────────┤
  │  Reason 3: Billing clarity                                 │
  │  Each API call is independent. You pay for tokens sent     │
  │  and received per call — clean accounting.                 │
  └────────────────────────────────────────────────────────────┘
```

The consequence: **memory is entirely your application's responsibility**.
The model cannot help you. It can only see what you send.

---

## Proof: The Stateless Call

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Call 1: introduce yourself
response_1 = llm.invoke([HumanMessage(content="Hi, my name is Alice.")])
print(response_1.content)
# "Hello Alice! Nice to meet you. How can I help you today?"

# Call 2: ask a follow-up question
response_2 = llm.invoke([HumanMessage(content="What's my name?")])
print(response_2.content)
# "I don't know your name — I don't have access to previous conversations."
# ↑ The model has NO idea who Alice is.
# Call 1 and Call 2 are completely independent from the model's perspective.
```

This is the root cause of the memory design problem.

---

## The Message List IS the Memory

The solution is straightforward: pass the conversation history as messages.

```python
# The model has access to anything in the message list.
# To give it "memory" of earlier turns, include those turns in the list.

response_2 = llm.invoke([
    HumanMessage(content="Hi, my name is Alice."),      # turn 1
    AIMessage(content="Hello Alice! Nice to meet you."),# turn 1 response
    HumanMessage(content="What's my name?"),            # turn 2
])
print(response_2.content)
# "Your name is Alice!"
```

The model did not gain a memory capability.
You passed the conversation history as part of the input.

```
Session state viewed by the model:

  ┌─────────────────────────────────────────────┐
  │ Call 2 input (what the model actually sees)  │
  │                                              │
  │  HumanMessage: "Hi, my name is Alice."       │ ← past turn
  │  AIMessage:    "Hello Alice! Nice to meet..."│ ← past turn
  │  HumanMessage: "What's my name?"             │ ← current turn
  └─────────────────────────────────────────────┘
                        │
                        ▼
            "Your name is Alice!"

  Your application is responsible for:
  1. Storing the past turns after each call
  2. Prepending them to every future call
```

---

## The Message List as a Data Structure

Every conversation is a list of typed messages.
Understanding the message types is essential:

```python
from langchain_core.messages import (
    SystemMessage,    # model behaviour instructions — usually first, usually constant
    HumanMessage,     # user's input
    AIMessage,        # model's response
    ToolMessage,      # result of a tool call (covered in Module 3)
)

# A typical conversation after 2 turns looks like:
messages = [
    SystemMessage(content="You are a helpful assistant."),   # constant
    HumanMessage(content="What is Python?"),                 # turn 1 user
    AIMessage(content="Python is a programming language."),  # turn 1 assistant
    HumanMessage(content="What version is current?"),        # turn 2 user
    AIMessage(content="Python 3.12 is the latest stable."),  # turn 2 assistant
    HumanMessage(content="When was 3.12 released?"),         # turn 3 user (current)
]

response = llm.invoke(messages)
# The model has full context of all prior turns — it can answer accurately.
```

---

## The Context Window Limit: Why You Can't Keep Everything

Every model has a **context window** — the maximum number of tokens it can process in one call.

```
Model context windows (approximate tokens):

  gpt-4o-mini:         128,000 tokens  ≈  96,000 words ≈  400 pages
  gpt-4o:              128,000 tokens
  claude-3.5-sonnet:   200,000 tokens  ≈ 150,000 words ≈  600 pages
  llama-3.2 (local):     8,000 tokens  ≈   6,000 words ≈   24 pages

A typical conversation turn: ~200 tokens (question + answer)
After 100 turns: 20,000 tokens
After 500 turns: 100,000 tokens  ← approaching gpt-4o-mini limit
```

Long conversations eventually hit the limit.
When they do, you must decide what to keep — that is the memory strategy problem.

---

## Why This Is the Memory Design Problem

The tension is between two constraints:

```
Constraint 1: To be contextually intelligent, the model needs past turns.
              More history → better responses.

Constraint 2: Context window is finite and tokens cost money.
              More history → higher cost, slower responses, eventual overflow.

              History items:    [H1] [A1] [H2] [A2] [H3] [A3] ... [H100] [A100]
              Context window:   |<─────────────────────── 128k tokens ──────────>|
                                              ↑
                                At some point this fills up.
                                You cannot simply append forever.
```

The three memory strategies (covered in Topic 02) are different answers
to the question: **which messages do you keep when the window fills up?**

---

## A Naive Manual Implementation

To make the model feel concrete, here is a minimal conversational loop with no framework:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# history holds all turns between calls
history: list = [
    SystemMessage(content="You are a helpful assistant.")
]

def chat(user_message: str) -> str:
    """Send a message and maintain conversation history."""
    history.append(HumanMessage(content=user_message))   # append user's turn

    response = llm.invoke(history)                        # send ALL history to model

    history.append(AIMessage(content=response.content))   # store model's response
    return response.content

# Test it
print(chat("My name is Alice."))    # "Hello Alice!"
print(chat("I work at TechCorp."))  # "That's great! TechCorp sounds interesting."
print(chat("What's my name?"))      # "Your name is Alice."
print(chat("Where do I work?"))     # "You work at TechCorp."

# The model remembers because history grows with each turn.
# This is the Full Buffer strategy — covered in Topic 02.
```

This works perfectly — until `history` grows large enough to hit the context window.

---

## What "Memory" Actually Means in LangChain

LangChain does not add memory to the model.
It provides utilities that automate:

1. Loading past messages from a storage backend.
2. Prepending them to the current invocation.
3. Saving the new turn back to the backend.

`RunnableWithMessageHistory` (Topic 03) is the abstraction that does this automatically.

```
Without RunnableWithMessageHistory:
  You manage the history list manually in every function.

With RunnableWithMessageHistory:
  LangChain manages loading, prepending, and saving automatically.
  You define: where to store (backend) and how to identify sessions (session_id).
```

---

## Common Pitfalls

| Pitfall                                          | What goes wrong                                                  | Fix                                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Assuming the model "remembers"                   | The model has no memory; it sees only what you send              | Always build history into your application layer                               |
| Using a global `history` list for multiple users | All users share one history; conversations bleed into each other | Use per-session history keyed by `session_id`                                  |
| Never trimming history                           | Context window overflow after long conversations                 | Implement a memory strategy (Topic 02)                                         |
| Adding a `SystemMessage` to history              | It gets repeated each turn, consuming tokens                     | Keep the system message separate; only include user/AI turns in stored history |
| Storing `AIMessage.content` as a string          | You lose message type metadata                                   | Store `AIMessage` objects, not `.content` strings                              |

---

## Mini Summary

- LLMs are stateless by design: every API call is independent.
- The message list IS the memory — the model has access to whatever you include in the call.
- The context window is finite; you cannot append conversation turns indefinitely.
- Memory is an **application responsibility** — the LLM never manages it.
- `RunnableWithMessageHistory` automates the load → prepend → save lifecycle.
- The question is not whether to add memory, but which strategy to use when the window fills.
