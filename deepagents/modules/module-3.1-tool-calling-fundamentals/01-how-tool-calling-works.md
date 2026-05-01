[← Module Overview](README.md) | [Next → Binding Tools](02-binding-tools.md)

---

# 01 — How Tool Calling Works

## Why This Matters First

Before you write a single line of `@tool`, you need to understand what actually happens
at the API boundary. A huge amount of confusion ("my tool is never called", "the model
hallucinated a tool name", "I got a ToolCall back but nothing executed") disappears the
moment you see what the model literally receives and what it literally sends back.

---

## Real-World Analogy

Imagine you're a new employee on your first day. Your manager gives you a printed sheet:

> **Available resources you may use:**
>
> ```
> lookup_policy(topic: str) -> str
>   "Search the employee policy handbook. topic is a keyword."
>
> book_meeting_room(room: str, time: str) -> bool
>   "Reserve a meeting room. room in ['A','B','C'], time in 'HH:MM' format."
> ```

When someone asks you a question, you decide: can I answer from memory, or do I need
to check the policy handbook first? If you need the handbook, you write a request slip:
`lookup_policy(topic="vacation days")`. An assistant fetches the answer and hands it back.
Only then do you compose your response.

The LLM does exactly this. The "printed sheet" is a JSON schema injected into the prompt.
The "request slip" is a `ToolCall` object in the response. The "assistant" is your application code.

---

## What the Model Actually Receives

When you call `.bind_tools(tools)` and then invoke the model, LangChain serializes each
tool into an OpenAI-compatible function schema and injects it into the API request:

```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the weather in Paris right now?" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Retrieve current weather for a city. Returns temperature in Celsius and condition.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "The city name, e.g. 'Paris' or 'New York'."
            }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

**Key insight:** the description field is part of the model's context window. It occupies
token space. A badly written description wastes tokens AND misleads the model.

---

## How the Model Signals a Tool Call

The model returns one of two response shapes:

### Shape 1 — Direct Answer (no tool needed)

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Paris is the capital of France.",
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Shape 2 — Tool Call Request

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"Paris\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

Notice:

- `finish_reason` changes from `"stop"` to `"tool_calls"` — this is the signal your app must check
- `arguments` is a **JSON string** (double-encoded), not an object — the model may produce invalid JSON
- `id` (`"call_abc123"`) must be echoed back in the `ToolMessage` so the model knows which call was answered

---

## The ToolCall Object in LangChain

LangChain parses the raw API response into a typed `AIMessage` with structured `tool_calls`:

```python
from langchain_core.messages import AIMessage, ToolCall

# What LangChain gives you after parsing the API response:
ai_msg = AIMessage(
    content="",   # often empty when the model chooses a tool
    tool_calls=[
        ToolCall(
            name="get_weather",
            args={"city": "Paris"},   # already parsed from JSON string
            id="call_abc123",
        )
    ]
)

# Access individual tool calls:
for tc in ai_msg.tool_calls:
    print(tc["name"])   # "get_weather"
    print(tc["args"])   # {"city": "Paris"}
    print(tc["id"])     # "call_abc123"
```

`ToolCall` is a TypedDict with three required keys: `name`, `args`, `id`.
There is also an optional `type` key (always `"tool_call"`).

---

## A Minimal End-to-End Demonstration

This example strips away all abstractions to show the raw mechanics:

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# ── 1. Define the tool ─────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Retrieve current weather for a city. Returns temperature in Celsius and condition."""
    # Stub — in production this would call a real weather API
    mock_data = {
        "Paris": "18°C, partly cloudy",
        "London": "12°C, rainy",
        "Tokyo": "25°C, sunny",
    }
    return mock_data.get(city, "Weather data unavailable for that city.")

# ── 2. Bind the tool to the model ──────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_weather])

# ── 3. First call: model decides to use the tool ───────────────────────────────
messages = [HumanMessage(content="What's the weather in Paris right now?")]
ai_response = llm_with_tools.invoke(messages)

print("finish_reason like signal:", ai_response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Paris'}, 'id': 'call_abc123', 'type': 'tool_call'}]

# ── 4. Execute the tool call ───────────────────────────────────────────────────
tool_call = ai_response.tool_calls[0]   # There may be multiple
tool_result = get_weather.invoke(tool_call["args"])

# ── 5. Feed the result back as a ToolMessage ───────────────────────────────────
tool_message = ToolMessage(
    content=tool_result,
    tool_call_id=tool_call["id"],   # MUST match the id from step 3
)

# ── 6. Second call: model reads the tool result and generates final answer ─────
messages = [*messages, ai_response, tool_message]
final_response = llm_with_tools.invoke(messages)

print(final_response.content)
# "The weather in Paris is currently 18°C and partly cloudy."
```

---

## Message List State After Each Step

```
Step 0:  [HumanMessage("What's the weather in Paris?")]

Step 1:  [HumanMessage(...),
          AIMessage(content="", tool_calls=[ToolCall(name="get_weather", args={"city":"Paris"}, id="call_abc123")])]

Step 2:  [HumanMessage(...),
          AIMessage(tool_calls=[...]),
          ToolMessage(content="18°C, partly cloudy", tool_call_id="call_abc123")]

Step 3:  [HumanMessage(...),
          AIMessage(tool_calls=[...]),
          ToolMessage(...),
          AIMessage(content="The weather in Paris is 18°C and partly cloudy.")]
```

The message list is a conversation _transcript_. Every participant writes into it, including tools.

---

## Under the Hood: Schema Generation

LangChain's `@tool` decorator inspects the function signature and docstring to build the JSON schema:

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="The city name, e.g. 'Paris'.")
    units: str = Field(default="celsius", description="'celsius' or 'fahrenheit'.")

@tool("get_weather", args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Retrieve current weather for a city."""
    ...

# Inspect what schema will be sent to the API:
import json
print(json.dumps(get_weather.args_schema.model_json_schema(), indent=2))
# {
#   "title": "WeatherInput",
#   "type": "object",
#   "properties": {
#     "city": {"title": "City", "description": "The city name, e.g. 'Paris'.", "type": "string"},
#     "units": {"title": "Units", "description": "'celsius' or 'fahrenheit'.", "default": "celsius", "type": "string"}
#   },
#   "required": ["city"]
# }
```

---

## How `finish_reason` Maps to LangChain Fields

| Raw API `finish_reason` | LangChain `AIMessage` signal                                     | Meaning                               |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------------- |
| `"stop"`                | `message.tool_calls == []`                                       | Model answered directly               |
| `"tool_calls"`          | `message.tool_calls != []`                                       | Model wants to call one or more tools |
| `"length"`              | `message.response_metadata["finish_reason"] == "length"`         | Max tokens exceeded mid-response      |
| `"content_filter"`      | `message.response_metadata["finish_reason"] == "content_filter"` | Response blocked                      |

---

## Common Pitfalls

| Pitfall                                                    | Symptom                                            | Fix                                                                            |
| ---------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------ |
| Not echoing `tool_call_id` in `ToolMessage`                | API error: "ToolMessage has no matching tool call" | Always use `tool_call_id=tool_call["id"]`                                      |
| Sending `ToolMessage` without the preceding `AIMessage`    | API 400 error                                      | The `AIMessage` with `tool_calls` must be in the list before the `ToolMessage` |
| Relying on `content` being non-empty to detect a tool call | Misses tool calls when content is `""`             | Check `len(ai_msg.tool_calls) > 0` instead                                     |
| Treating `args` as a JSON string                           | `KeyError` or wrong types                          | LangChain parses `arguments` into a dict automatically                         |
| Assuming one tool call per response                        | Second tool silently ignored                       | Always loop over `ai_msg.tool_calls`, not just `[0]`                           |

---

## Mini Summary

- Tool calling works by injecting JSON function schemas into the API request as part of the context window
- The model signals intent via `finish_reason: "tool_calls"` and populates `tool_calls` in the response
- LangChain parses this into `AIMessage.tool_calls` — a list of `ToolCall` TypedDicts with `name`, `args`, `id`
- Your app executes the tool and returns a `ToolMessage` with matching `tool_call_id`
- The model sees the result and generates a final answer on the next call
- Message ordering is mandatory: `AIMessage(tool_calls)` → `ToolMessage` → next LLM call

---

[← Module Overview](README.md) | [Next → Binding Tools](02-binding-tools.md)
