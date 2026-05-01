# Module 3.2 — Custom Tools

> **Why this module exists:** The built-in tools you find in LangChain Community cover
> common integrations, but production agents always need domain-specific tools — tools
> that know about your database schema, your internal API, your proprietary data format.
> This module teaches you to build, test, and manage tools that are yours.

---

## Topics

| #   | File                                                          | What you will learn                                                          |
| --- | ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 01  | [The @tool Decorator](01-tool-decorator.md)                   | Deep dive into `@tool`, Pydantic v2 schemas, docstring quality, return types |
| 02  | [Async Tools](02-async-tools.md)                              | `async def` tools, why async matters in agents, event loop in LangGraph      |
| 03  | [Tool Exception Patterns](03-tool-exception-patterns.md)      | `ToolException`, callbacks, partial results, the safe-tool pattern           |
| 04  | [Testing Tools in Isolation](04-tool-testing-in-isolation.md) | pytest patterns, mocking APIs, input validation tests, error path tests      |
| 05  | [Tool Registry Pattern](05-tool-registry-pattern.md)          | Managing tool libraries, role-based grouping, runtime selection, versioning  |

---

## Module Architecture — `@tool` Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     @tool DECORATOR LIFECYCLE                                │
│                                                                              │
│  ① DEFINITION TIME                                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  @tool                                                                │   │
│  │  def my_tool(x: str) -> str: ...                                      │   │
│  │          │                                                             │   │
│  │          ▼                                                             │   │
│  │   Pydantic schema built from type hints + docstring                   │   │
│  │   StructuredTool object created with .name, .description, .invoke()  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ② BINDING TIME                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  llm.bind_tools([my_tool])                                            │   │
│  │          │                                                             │   │
│  │          ▼                                                             │   │
│  │   tool.args_schema.model_json_schema() → OpenAI function object       │   │
│  │   Injected into every API call as "tools": [...]                      │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ③ INVOCATION TIME                                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  model returns ToolCall(name="my_tool", args={"x": "value"})          │   │
│  │          │                                                             │   │
│  │          ▼                                                             │   │
│  │  tool.invoke({"x": "value"})                                          │   │
│  │          │                                                             │   │
│  │          ▼                                                             │   │
│  │  Pydantic validates args → function called → result returned          │   │
│  │  ToolMessage(content=result, tool_call_id=id)                         │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Module 3.1 (tool calling mechanics)
- Pydantic v2 basics (BaseModel, Field)
- Python `async/await` basics

## Key Packages

```
langchain-core>=0.3
pydantic>=2.0
aiohttp>=3.9     # for async tool examples
pytest>=8.0      # for testing examples
pytest-asyncio   # for async tool tests
```
