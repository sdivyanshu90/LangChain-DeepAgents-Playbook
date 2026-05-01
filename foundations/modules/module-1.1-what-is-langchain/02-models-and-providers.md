# 02 — Models and Providers

> **Previous:** [01 → Why LangChain Exists](01-why-langchain-exists.md) | **Next:** [03 → Prompts and Messages](03-prompts-and-messages.md)

---

## Real-World Analogy

A car manufacturer's engine is the reasoning core.
The steering wheel, pedals, and dashboard are the interface your code uses.

`ChatOpenAI` and `ChatAnthropic` are those steering wheels.
They give you a standard interface so the engine (GPT-4, Claude, Gemini) can be swapped without rewiring the entire car.

---

## What a Chat Model Actually Does

Before writing any code, understand the underlying contract:

```
INPUT:  A list of typed messages
           │
           │  [SystemMessage("You are..."), HumanMessage("Tell me about...")]
           │
           ▼
    ┌─────────────┐
    │  Chat Model │  (stateless — each call is independent)
    └──────┬──────┘
           │
           ▼
OUTPUT: A single AIMessage
           │
           │  AIMessage(content="...", response_metadata={...})
```

Critical point: **the model has no memory between calls**.
Everything the model needs to know must be in the input message list.
Memory is always an application-level concern.

---

## ChatOpenAI — In Depth

### Initialization

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env

llm = ChatOpenAI(
    model="gpt-4o-mini",      # model name — never hardcode, use env var
    temperature=0,             # 0 = deterministic; higher = more creative
    max_tokens=1024,           # hard cap on output length
    timeout=30,                # seconds before raising a timeout error
    max_retries=2,             # automatic retry on transient failures
)
```

### Key Parameters Explained

| Parameter      | Type        | When to use                                       | Why it matters                                                       |
| -------------- | ----------- | ------------------------------------------------- | -------------------------------------------------------------------- |
| `temperature`  | `float` 0–2 | 0 for agents/extraction; 0.7–1 for creative tasks | Controls output randomness. Agents need 0 for deterministic routing. |
| `max_tokens`   | `int`       | Always set in production                          | Prevents runaway cost on long prompts                                |
| `timeout`      | `int`       | Any network-dependent code                        | Prevents silent hangs in production                                  |
| `max_retries`  | `int`       | Production; rate-limited APIs                     | Handles transient 429/500 errors automatically                       |
| `streaming`    | `bool`      | UI-facing applications                            | Enables token-by-token streaming for better UX                       |
| `model_kwargs` | `dict`      | Advanced: seed, response_format                   | Pass provider-specific params not yet in the SDK                     |

### Response Metadata

Every call returns an `AIMessage` with attached metadata:

```python
response = llm.invoke([HumanMessage(content="What is 2+2?")])

print(response.content)           # "4"
print(response.response_metadata) # token counts, model name, finish reason
# {'token_usage': {'prompt_tokens': 13, 'completion_tokens': 1, 'total_tokens': 14},
#  'model_name': 'gpt-4o-mini', 'finish_reason': 'stop'}
```

This is how you track actual cost and latency per call.

---

## ChatAnthropic — In Depth

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=1024,
    # Anthropic requires max_tokens — there is no default
)
```

### OpenAI vs Anthropic — Key Differences

```
┌────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Dimension              │ OpenAI (GPT-4o)           │ Anthropic (Claude 3.5)   │
├────────────────────────┼──────────────────────────┼──────────────────────────┤
│ max_tokens default     │ model-dependent           │ REQUIRED — must be set   │
│ System message support │ Yes (role="system")       │ Yes (role="system")      │
│ Tool calling           │ Yes                       │ Yes                      │
│ Streaming              │ Yes                       │ Yes                      │
│ Response format        │ json_object supported     │ Use structured output    │
│ Context window         │ 128k (gpt-4o)             │ 200k (claude-3.5-sonnet) │
└────────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## Provider-Agnostic Code with model_factory

The best pattern: never instantiate a model directly in application code.
Use a factory that reads the provider from environment variables:

```python
# shared/utils/model_factory.py  (already in this repo)
from shared.utils.model_factory import get_chat_model

# Switch provider by changing MODEL_PROVIDER= in .env
llm = get_chat_model(temperature=0)
```

This means you can run the same project against GPT-4o, Claude, or Gemini
by changing one line in `.env`, with zero code changes.

---

## Invoking a Model — Four Modes

### 1. Synchronous (default)

```python
response = llm.invoke([HumanMessage(content="Explain embeddings.")])
print(response.content)
```

### 2. Streaming (token by token)

```python
# Ideal for Streamlit, FastAPI SSE, CLI UX
for chunk in llm.stream([HumanMessage(content="Explain embeddings.")]):
    print(chunk.content, end="", flush=True)
```

### 3. Batch (multiple inputs at once)

```python
# More efficient than calling invoke() in a loop
responses = llm.batch([
    [HumanMessage(content="Explain embeddings.")],
    [HumanMessage(content="Explain vector stores.")],
])
```

### 4. Async

```python
import asyncio

async def main():
    response = await llm.ainvoke([HumanMessage(content="Explain embeddings.")])
    print(response.content)

asyncio.run(main())
```

### When to Use Which

```
invoke()   → scripts, synchronous pipelines, CLI tools
stream()   → any UI that should show partial output progressively
batch()    → bulk processing (documents, records) — use over a loop
ainvoke()  → async web frameworks (FastAPI, Starlette), concurrent agents
```

---

## Token Counting Before the Call

Overfilling the context window raises an error. Check before you call:

```python
# Count tokens without making an API call
token_count = llm.get_num_tokens("Your text here")
print(f"This input uses {token_count} tokens")

# Check against the model's limit
MAX_TOKENS = 128_000  # gpt-4o
if token_count > MAX_TOKENS * 0.8:  # leave 20% for the response
    raise ValueError(f"Input too long: {token_count} tokens")
```

---

## Common Pitfalls

| Pitfall                                | What happens                         | Fix                                         |
| -------------------------------------- | ------------------------------------ | ------------------------------------------- |
| Hardcoding the API key                 | Key exposed in git history           | Always use `python-dotenv` + `.env`         |
| Not setting `max_tokens` in production | Runaway costs on verbose completions | Always set an explicit upper bound          |
| Using `temperature=1` in agents        | Non-deterministic routing decisions  | Use `temperature=0` for all agent nodes     |
| Calling `invoke()` in a loop           | N sequential API round-trips         | Use `batch()` for parallelism               |
| Not reading `response_metadata`        | Token costs invisible                | Log `response_metadata` for every prod call |
| Treating `AIMessage` as a string       | `TypeError` downstream               | Always call `.content` or use a parser      |

---

## Mini Summary

- Chat models are stateless — the entire context is the input message list.
- `ChatOpenAI` and `ChatAnthropic` share the same interface; swapping is a config change.
- Always set `temperature=0` for agents, `max_tokens` for cost control, `timeout` for production.
- Use `batch()` instead of a loop; use `stream()` for any UI-facing output.
- Never hardcode API keys — use `python-dotenv`.

---

## Next: [03 → Prompts and Messages](03-prompts-and-messages.md)
