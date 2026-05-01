# Module 1.1 — Models, Prompts, and Parsers

> **DeepAgent Level: 0 → 1** | Estimated reading time: 40 min | Prerequisites: Python basics, pip, an OpenAI API key

This module covers the atomic primitives of every LangChain application.
Everything built in Level 2 and Level 3 is composed from exactly these pieces.

---

## Why Start Here?

Most developers jump straight to building a chatbot and wonder why it breaks.
The reason is almost always a shaky understanding of how messages, models, and parsers work together.

Before writing a single RAG pipeline or agent graph, you must be able to answer:

- How does a chat model actually receive instructions?
- What is the difference between a prompt template and a raw string?
- Why does the output need parsing at all?

This module answers all three.

---

## Topics in This Module

| # | File | What you will learn |
|---|---|---|
| 01 | [Why LangChain Exists](01-why-langchain-exists.md) | The problems LangChain solves; the abstraction ladder from raw API to composable pipeline |
| 02 | [Models and Providers](02-models-and-providers.md) | ChatOpenAI, ChatAnthropic, model parameters, provider-agnostic code |
| 03 | [Prompts and Messages](03-prompts-and-messages.md) | ChatPromptTemplate, SystemMessage, HumanMessage, AIMessage, message roles |
| 04 | [Output Parsers](04-output-parsers.md) | StrOutputParser, JsonOutputParser, PydanticOutputParser — what each does and when |
| 05 | [Your First Complete Chain](05-first-complete-chain.md) | Putting all four primitives together with full inline commentary |

---

## The Mental Model

```
User Input
    │
    ▼
┌─────────────────────────┐
│   ChatPromptTemplate    │  ← defines the message structure
│  (template → messages)  │
└──────────┬──────────────┘
           │ list[BaseMessage]
           ▼
┌─────────────────────────┐
│      Chat Model         │  ← reasons over the messages
│  (messages → AIMessage) │
└──────────┬──────────────┘
           │ AIMessage
           ▼
┌─────────────────────────┐
│     Output Parser       │  ← converts model output → app data
│   (AIMessage → result)  │
└──────────┬──────────────┘
           │ str | dict | Pydantic model
           ▼
   Application Output
```

Every LangChain system — from a simple one-shot chain to a 12-node DeepAgent —
is built on this same three-stage structure.

---

## Start With: [01 → Why LangChain Exists](01-why-langchain-exists.md)
