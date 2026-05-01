# 02 — Runnable Types Deep Dive

> **Previous:** [01 → The Pipe Operator](01-pipe-operator-and-data-flow.md) | **Next:** [03 → Streaming and Batching](03-streaming-and-batching.md)

---

## The Runnable Family

```
Runnable (base interface)
├── RunnableSequence     — A | B | C  (sequential pipeline)
├── RunnableParallel     — {key1: A, key2: B}  (fan-out, same input → multiple outputs)
├── RunnablePassthrough  — passes input unchanged (with optional .assign())
├── RunnableLambda       — wraps any Python callable as a Runnable
└── RunnableBranch       — conditional routing based on a predicate
```

Each type solves a different composition problem.
Most complex LangChain chains are combinations of just these five.

---

## RunnableSequence

Already covered in Topic 01. Quick reference:

```python
from langchain_core.runnables import RunnableSequence

# Both are equivalent:
chain = prompt | model | parser
chain = RunnableSequence(prompt, model, parser)

# Invoke, stream, batch all work on the composed sequence
result = chain.invoke({"question": "..."})
```

---

## RunnableParallel — Fan-Out

The single most useful combinator for building data-rich responses.

**One input feeds multiple independent branches simultaneously.**
Each branch produces a separate output key in a dict.

```
                    {"text": "..."}
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    summary_chain   risks_chain   keywords_chain
          │               │               │
          ▼               ▼               ▼
     "summary..."    ["risk 1",...]   ["python",...]
          │               │               │
          └───────────────┼───────────────┘
                          │
          {"summary": "...", "risks": [...], "keywords": [...]}
```

### Code

```python
from langchain_core.runnables import RunnableParallel

summary_prompt  = ChatPromptTemplate.from_messages([
    ("system", "Summarise in one sentence."),
    ("human",  "{text}"),
])
risks_prompt = ChatPromptTemplate.from_messages([
    ("system", "List the top 3 risks as a JSON array."),
    ("human",  "{text}"),
])

analysis = RunnableParallel(
    summary  = summary_prompt  | llm | StrOutputParser(),
    risks    = risks_prompt    | llm | JsonOutputParser(),
    original = RunnablePassthrough(),   # pass input unchanged
)

result = analysis.invoke({"text": "Deploying AI to production has several risks..."})
# {
#   "summary": "Deploying AI requires careful risk management.",
#   "risks": ["data drift", "latency spikes", "hallucinations"],
#   "original": {"text": "Deploying AI to production..."}
# }
```

### Why Parallel, Not Sequential?

```
Sequential (slow):   A → B → C  =  300ms + 280ms + 320ms = 900ms
Parallel (fast):     A           = 300ms
                       B         = 280ms  (concurrent)
                         C       = 320ms
                     Total: max(300, 280, 320) = 320ms
```

`RunnableParallel` calls all branches concurrently using `asyncio` when available.
For N independent operations, it saves roughly `(N-1) × avg_latency` in wall-clock time.

---

## RunnablePassthrough — Preserve Input

Passes the input unchanged. Useful when you need the original value alongside derived values.

```python
from langchain_core.runnables import RunnablePassthrough

# Pass the question through unchanged alongside the generated answer
chain = RunnableParallel(
    question = RunnablePassthrough(),
    answer   = prompt | llm | StrOutputParser(),
)

result = chain.invoke({"question": "What is LCEL?"})
# {"question": {"question": "What is LCEL?"}, "answer": "LCEL is..."}
```

### `.assign()` — Add Fields Without Replacing the Input

The most common use: incrementally enrich a dict as it flows through the pipeline.

```python
from langchain_core.runnables import RunnablePassthrough

# Each .assign() adds new keys to the flowing dict
enrichment_chain = (
    RunnablePassthrough.assign(
        summary=lambda x: summarize(x["text"])
    )
    .assign(
        word_count=lambda x: len(x["text"].split())
    )
    .assign(
        is_long=lambda x: x["word_count"] > 500
    )
)

result = enrichment_chain.invoke({"text": "..."})
# {"text": "...", "summary": "...", "word_count": 342, "is_long": False}
```

This pattern is the standard way to build sequential enrichment pipelines
without losing earlier fields.

---

## RunnableLambda — Wrap Any Python Function

When you need a plain function to participate in a LCEL pipeline:

````python
from langchain_core.runnables import RunnableLambda

def clean_text(text: str) -> str:
    """Remove markdown code fences from model output."""
    return text.strip().removeprefix("```").removesuffix("```").strip()

# Wrap as a Runnable so it can be piped
clean = RunnableLambda(clean_text)

chain = prompt | llm | StrOutputParser() | clean

# Equivalent shorthand — functions are auto-wrapped when used in a pipe
chain = prompt | llm | StrOutputParser() | clean_text   # also works
````

### Async Lambda

For async operations (HTTP calls, DB lookups):

```python
import asyncio

async def fetch_related(query: str) -> list[str]:
    await asyncio.sleep(0)  # placeholder for async I/O
    return [f"related: {query}"]

chain = prompt | llm | StrOutputParser() | RunnableLambda(fetch_related)
```

---

## RunnableBranch — Conditional Routing

Routes to different sub-chains based on a predicate evaluated against the current input.

```python
from langchain_core.runnables import RunnableBranch

# Each tuple: (predicate_function, chain_if_true)
# Last argument: default chain (no predicate)

branch = RunnableBranch(
    (
        lambda x: x["language"] == "es",
        spanish_chain,          # invoked if language is Spanish
    ),
    (
        lambda x: x["language"] == "fr",
        french_chain,           # invoked if language is French
    ),
    english_chain,              # default
)

result = branch.invoke({"text": "Hola mundo", "language": "es"})
# → routed to spanish_chain
```

### When to Use RunnableBranch vs LangGraph

```
RunnableBranch is appropriate when:
  ✓ The branching condition is simple (one field check)
  ✓ The branches are stateless and do not accumulate results
  ✓ You are working at Level 1 (chains, not agents)

Use LangGraph conditional_edges instead when:
  ✓ The routing involves multiple fields or complex logic
  ✓ Branches may loop back
  ✓ State must persist across the branch
  ✓ You are at Level 3+ (stateful agents)
```

---

## Combining All Five — A Real Example

A research chain that:

1. Detects the question language (RunnableLambda)
2. Runs summary + key facts in parallel (RunnableParallel)
3. Routes to a language-appropriate formatter (RunnableBranch)
4. Preserves the original input in the output (RunnablePassthrough)

```python
chain = (
    RunnablePassthrough.assign(
        language=RunnableLambda(detect_language)
    )
    .assign(
        analysis=RunnableParallel(
            summary=summary_chain,
            facts=facts_chain,
        )
    )
    | RunnableBranch(
        (lambda x: x["language"] == "es", spanish_formatter),
        english_formatter,
    )
)
```

```
Input: {"question": "¿Qué es LCEL?"}
│
├─ assign language → "es"
│
├─ assign analysis → {summary: "...", facts: [...]}  (parallel)
│
└─ RunnableBranch → spanish_formatter (language == "es")
        │
        ▼
"LCEL es un lenguaje de composición para LangChain..."
```

---

## Common Pitfalls

| Pitfall                                              | What breaks                          | Fix                                                                                         |
| ---------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `RunnableParallel` with dependent branches           | Second branch receives stale data    | Use `.assign()` for sequential enrichment, `RunnableParallel` only for independent branches |
| `RunnableLambda` with a function that returns `None` | `TypeError` in the next stage        | Every lambda must return a value                                                            |
| `RunnableBranch` without a default                   | `ValueError` if no predicate matches | Always include a default chain as the final argument                                        |
| Mutable state inside a lambda                        | Unpredictable results in `.batch()`  | Keep lambdas pure — no side effects, no shared mutable state                                |

---

## Mini Summary

- `RunnableSequence` (`|`): A → B → C, output of A feeds B, output of B feeds C.
- `RunnableParallel`: same input fans out to N branches concurrently; outputs merged into a dict.
- `RunnablePassthrough`: passes input unchanged; `.assign()` adds fields without replacing.
- `RunnableLambda`: wraps any Python function as a first-class Runnable.
- `RunnableBranch`: conditional routing; use at Level 1; prefer LangGraph at Level 3+.

---

## Next: [03 → Streaming and Batching](03-streaming-and-batching.md)
