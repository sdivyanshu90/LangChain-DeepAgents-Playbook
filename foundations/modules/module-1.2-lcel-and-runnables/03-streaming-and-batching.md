# 03 — Streaming and Batching

> **Previous:** [02 → Runnable Types](02-runnable-types.md) | **Next:** [04 → Data Shape Discipline](04-data-shape-discipline.md)

---

## Why These Two Modes Matter in Production

Most tutorials show `chain.invoke()` and stop.
In production you almost always need either:

- **Streaming** — for UI-facing features where users should see tokens arrive progressively
- **Batching** — for processing lists of inputs efficiently without N sequential API calls

Both are first-class features of every Runnable, not bolt-on extras.

---

## Streaming — Token by Token

### Why Streaming Exists

Without streaming, a user stares at a blank screen for 2–10 seconds, then the full response appears.
With streaming, the first token appears in ~200ms and the response unfolds in real time.

This is a significant UX difference, especially for long answers.

### `.stream()` — Synchronous

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
chain = prompt | llm | StrOutputParser()

# Stream tokens as they arrive
for chunk in chain.stream({"question": "Explain embeddings in 3 paragraphs."}):
    print(chunk, end="", flush=True)
print()  # newline at end
```

**What `chunk` is:**  
After `StrOutputParser`, each chunk is a `str` — a small piece of the response text.
Before the parser, each chunk from `ChatOpenAI` is an `AIMessageChunk` with a `.content` field.

### `.astream()` — Async

Required in async web frameworks (FastAPI, Starlette):

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat")
async def chat(question: str):
    async def token_generator():
        async for chunk in chain.astream({"question": question}):
            yield chunk

    return StreamingResponse(token_generator(), media_type="text/plain")
```

### Streaming with Structured Output

When using `.with_structured_output()`, streaming returns partial Pydantic objects as they fill in:

```python
structured_llm = llm.with_structured_output(ContactCard)
chain = prompt | structured_llm

for partial in chain.stream({"input": text}):
    # partial is a partial ContactCard — some fields may be None still
    print(partial)
# ContactCard(name='Alex', email=None, company=None, ...)
# ContactCard(name='Alex', email='alex@acme.com', company=None, ...)
# ContactCard(name='Alex', email='alex@acme.com', company='Acme', ...)
```

Useful for showing progressive field extraction in a UI.

---

## Streaming in a Streamlit App

```python
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | llm | StrOutputParser()

question = st.text_input("Ask anything:")
if question:
    # st.write_stream consumes the generator and displays tokens live
    response = st.write_stream(chain.stream({"q": question}))
```

The key: pass `streaming=True` at model init OR rely on `.stream()` at call time — both work.

---

## Batching — Parallel Processing

### Why Batching Exists

```
# ❌ The naive loop — N sequential API round-trips
results = []
for text in documents:
    results.append(chain.invoke({"text": text}))
# Total time: N × avg_latency

# ✅ Batch — concurrent API calls
results = chain.batch([{"text": t} for t in documents])
# Total time: max(latency per call) ≈ avg_latency (N calls run in parallel)
```

For 20 documents at 600ms each:

- Loop: `20 × 600ms = 12 seconds`
- Batch: `≈ 600ms` (all concurrent, with provider rate limit headroom)

### `.batch()` Basic Usage

```python
inputs = [
    {"question": "What is LCEL?"},
    {"question": "What is LangGraph?"},
    {"question": "What is a vector store?"},
]

results = chain.batch(inputs)
# Returns a list in the same order as inputs
for q, r in zip(inputs, results):
    print(f"Q: {q['question']}")
    print(f"A: {r}\n")
```

### Controlling Concurrency

By default, `.batch()` is fully concurrent. Use `max_concurrency` to avoid rate limiting:

```python
results = chain.batch(
    inputs,
    config={"max_concurrency": 5},  # at most 5 concurrent API calls
)
```

### `.abatch()` — Async Batch

```python
import asyncio

async def process_all(texts: list[str]) -> list[str]:
    inputs = [{"text": t} for t in texts]
    return await chain.abatch(inputs, config={"max_concurrency": 10})

results = asyncio.run(process_all(documents))
```

---

## Streaming vs Batching — Decision Guide

```
User is waiting for output in a UI?
    └── Yes → stream()  or  astream()

Processing multiple inputs at once?
    └── Yes → batch()  or  abatch()

Single input, synchronous script?
    └── Yes → invoke()

Single input, async app?
    └── Yes → ainvoke()

Streaming multiple inputs at once?
    └── astream_log() or astream_events() — advanced, see LangSmith docs
```

---

## `astream_events` — Introspecting Mid-Stream

For complex chains (RAG, agents), you may want to intercept events from specific nodes:

```python
async def monitor_chain():
    async for event in chain.astream_events(
        {"question": "What is LCEL?"},
        version="v2",
    ):
        # Filter to model events only
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif event["event"] == "on_chain_end":
            print(f"\n[Chain finished: {event['name']}]")
```

This is the foundation of real-time streaming UIs that also show intermediate steps
(like "Searching documents…", "Generating answer…").

---

## Common Pitfalls

| Pitfall                                      | What breaks                                     | Fix                                                             |
| -------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------- |
| Using `invoke()` in a loop over 50 documents | ~30 second runtime                              | Use `batch()` with `max_concurrency`                            |
| No `max_concurrency` on large batches        | Provider rate-limit 429 errors                  | Set `max_concurrency=5` to start; tune up from there            |
| Forgetting `flush=True` in `stream()` print  | No visible progress in terminal                 | Always `print(chunk, end="", flush=True)`                       |
| Using `stream()` with `PydanticOutputParser` | No partial objects — parser waits for full text | Use `.with_structured_output()` for streaming structured output |

---

## Mini Summary

- `.stream()` / `.astream()` deliver tokens progressively — essential for any user-facing output.
- `.batch()` / `.abatch()` run N inputs concurrently — use instead of a loop whenever processing lists.
- `max_concurrency` prevents rate-limit errors on large batches.
- `astream_events` intercepts specific node events — the foundation of rich streaming UIs.

---

## Next: [04 → Data Shape Discipline](04-data-shape-discipline.md)
