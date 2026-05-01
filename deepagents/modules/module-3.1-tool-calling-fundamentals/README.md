# Module 3.1 — Tool Calling Fundamentals

> **Why this module exists:** An LLM with no tools can only reason about things it has already seen.
> Tool calling is the mechanism that lets a model reach _outside its context window_ into the real world —
> querying live data, writing files, executing code, or calling APIs — and then reason about the results.

---

## Topics

| #   | File                                                   | What you will learn                                                                                              |
| --- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 01  | [How Tool Calling Works](01-how-tool-calling-works.md) | API-level mechanics: JSON schemas injected into the prompt, how the model signals a tool call vs a direct answer |
| 02  | [Binding Tools](02-binding-tools.md)                   | `.bind_tools()`, `tool_choice` modes, parallel tool calls, inspecting bound tools                                |
| 03  | [Tool Message Parsing](03-tool-message-parsing.md)     | `ToolCall` vs `ToolMessage`; the two-step loop; why message ordering matters                                     |
| 04  | [Tool Error Handling](04-tool-error-handling.md)       | `ToolException`, `handle_tool_error`, graceful degradation, structured error responses                           |
| 05  | [When to Use Tools](05-when-to-use-tools.md)           | Decision framework: tool vs prompt engineering vs chain; scope, naming, side-effects                             |

---

## Module Architecture — The Agent Decision Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENT DECISION LOOP                              │
│                                                                         │
│   ┌──────────┐      ┌─────────────────┐      ┌──────────────────────┐  │
│   │          │      │                 │      │                      │  │
│   │  THINK   │─────▶│  CHOOSE TOOL?   │─────▶│   EXECUTE TOOL       │  │
│   │          │      │                 │  YES │   (external world)   │  │
│   │ (LLM     │      │ model emits     │      │                      │  │
│   │  reasons │      │ ToolCall object │      │   tool returns       │  │
│   │  over    │      │ or direct text) │      │   ToolMessage        │  │
│   │  context)│      │                 │      │                      │  │
│   └──────────┘      └────────┬────────┘      └──────────┬───────────┘  │
│         ▲                    │ NO                        │              │
│         │                    ▼                           │              │
│         │            ┌───────────────┐                  │              │
│         │            │   RESPOND     │     OBSERVE       │              │
│         │            │   directly    │◀──────────────────┘              │
│         │            │   to user     │  (ToolMessage appended           │
│         │            └───────────────┘   to message list, loop back)   │
│         │                                                               │
│         └────────────────────────────────────────────────────────────  │
│                       next iteration                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Message list grows with each iteration

```
[SystemMessage]
[HumanMessage: "What's the weather in Paris?"]
  ──▶ LLM emits AIMessage(tool_calls=[ToolCall(name="get_weather", args={"city":"Paris"})])
[AIMessage with tool_calls]
[ToolMessage: '{"temp": 18, "condition": "partly cloudy"}']
  ──▶ LLM emits AIMessage(content="It is 18°C and partly cloudy in Paris.")
[AIMessage: final answer]
```

---

## Prerequisites

- Module 1 (LCEL, Runnables, output formatting)
- Module 2 (memory, message history)
- Familiarity with OpenAI chat API request/response structure

## Key Packages

```
langchain-core>=0.3
langchain-openai>=0.2
```
