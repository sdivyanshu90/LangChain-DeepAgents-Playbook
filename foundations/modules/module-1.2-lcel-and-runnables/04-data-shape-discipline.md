# 04 — Data Shape Discipline

> **Previous:** [03 → Streaming and Batching](03-streaming-and-batching.md) | **Next:** [05 → Composition Patterns](05-composition-patterns.md)

---

## Real-World Analogy

Imagine a factory assembly line where each station passes a tray to the next.
Station 2 always expects a tray with one bolt and two washers.
If Station 1 passes a tray with three bolts and no washers, Station 2 jams.

LCEL chains have the same problem.
Every stage has an expected input shape.
If the previous stage produces a different shape, the pipeline breaks — often with a confusing error.

**Data shape discipline** is the skill of knowing, at every point in your pipeline,
what type and structure the current value has.

---

## The Four Common Shape Errors

```
Error 1 — Wrong type:
  stage receives `str`, but expected `dict`
  → TypeError or KeyError

Error 2 — Missing key:
  stage receives {"answer": "..."}, but tries to access x["question"]
  → KeyError

Error 3 — Extra/unexpected nesting:
  stage receives {"result": {"answer": "..."}}
  but tries to access x["answer"]
  → KeyError

Error 4 — List where scalar expected:
  stage receives [AIMessage(...)], but str(message) is called directly
  → Produces garbage output, not an exception
```

---

## Tracing Shapes Through a Chain

The best debugging habit: annotate the output type of every stage.

```python
chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    # OUT: list[BaseMessage]
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # OUT: AIMessage
    | StrOutputParser()
    # OUT: str
)
```

When you add a step, ask: "What does the previous stage produce, and does my new step accept that?"

---

## Using `.assign()` to Build Up a Dict

The most common shape-management pattern:
start with the original dict and accumulate new keys.

```python
from langchain_core.runnables import RunnablePassthrough

# At each .assign() step, the input dict grows with a new key
# All prior keys remain available

pipeline = (
    RunnablePassthrough.assign(
        # Input:  {"text": "..."}
        # Output: {"text": "...", "char_count": 342}
        char_count=lambda x: len(x["text"])
    )
    .assign(
        # Input:  {"text": "...", "char_count": 342}
        # Output: {"text": "...", "char_count": 342, "is_long": True}
        is_long=lambda x: x["char_count"] > 300
    )
    .assign(
        # Input:  {"text": "...", "char_count": 342, "is_long": True}
        # Output: adds "summary" key
        summary=summary_chain
        # summary_chain.invoke receives the full dict {"text": ..., "char_count": ..., "is_long": ...}
        # It uses x["text"] internally
    )
)
```

### When a sub-chain needs only one key

Use `RunnableLambda` to extract before passing to a chain that expects a dict:

```python
# summary_chain expects {"text": "..."}  (just the text key)
# But our accumulated dict has many keys

pipeline = (
    RunnablePassthrough.assign(
        summary=RunnableLambda(lambda x: {"text": x["text"]}) | summary_chain
    )
)
```

---

## Itemgetter — Extracting a Single Value

When a chain needs a scalar string but the flowing value is a dict:

```python
from operator import itemgetter

chain = (
    # Input: {"question": "...", "context": "..."}
    {
        "question": itemgetter("question"),
        "context":  itemgetter("context"),
    }
    | answer_prompt    # expects dict with "question" and "context"
    | llm
    | StrOutputParser()
)
```

`itemgetter("question")` is equivalent to `lambda x: x["question"]` but is serializable
(important for LangSmith tracing and graph visualization).

---

## Debugging Shape Problems

### Step 1 — Add a peek lambda

```python
def peek(x):
    """Inspect the flowing value without changing it."""
    import json
    try:
        print(f"\n[PEEK] type={type(x).__name__}  value={json.dumps(x, default=str)[:200]}")
    except Exception:
        print(f"\n[PEEK] type={type(x).__name__}  value={repr(x)[:200]}")
    return x  # must return the input unchanged

# Insert at any point in the chain
chain = prompt | llm | RunnableLambda(peek) | parser
```

### Step 2 — Use `.invoke()` on sub-chains

```python
# Don't run the whole chain — run each stage independently to find the break
messages = prompt.invoke({"question": "test"})
print(type(messages), messages)

response = llm.invoke(messages)
print(type(response), response)

result = parser.invoke(response)
print(type(result), result)
```

### Step 3 — Read LangSmith trace

When tracing is enabled, LangSmith shows the exact input and output at every stage.
This is the fastest way to spot a shape mismatch in a complex multi-stage pipeline.

---

## Common Shape Patterns Reference

```
Pattern                              Input shape         Output shape
─────────────────────────────────────────────────────────────────────
prompt | llm | StrOutputParser()     dict                str
prompt | llm | JsonOutputParser()    dict                dict
prompt | structured_llm              dict                BaseModel
RunnableParallel({a: chainA,         any                 {"a": ..., "b": ...}
                 b: chainB})
RunnablePassthrough.assign(k=f)      dict                dict (+ new key k)
RunnableLambda(fn)                   any                 return type of fn
itemgetter("key")                    dict                dict["key"]
```

---

## Common Pitfalls

| Pitfall                                                                      | Error message                         | Fix                                                                   |
| ---------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------- |
| `prompt.invoke("string")` instead of `prompt.invoke({"question": "string"})` | `KeyError` or `TypeError`             | Prompts always take `dict` with named variables                       |
| Chaining after `StrOutputParser` into a prompt                               | `TypeError: expected dict, got str`   | Wrap in `{"variable": RunnablePassthrough()}` or use `RunnableLambda` |
| Forgetting `.assign()` returns the full dict                                 | Wondering where prior keys went       | They are all still there — check with `peek`                          |
| `RunnableParallel` sub-chain that modifies global state                      | Unpredictable concurrent side effects | Keep sub-chains pure                                                  |

---

## Mini Summary

- Every stage in a LCEL chain has an expected input type. Track shapes explicitly.
- `.assign()` is the standard way to accumulate keys in a flowing dict without losing prior data.
- `itemgetter` cleanly extracts a single key when a downstream chain needs a scalar.
- A `peek` lambda is the fastest debugging tool for shape problems.
- LangSmith trace shows the exact input/output at every boundary — use it.

---

## Next: [05 → Composition Patterns](05-composition-patterns.md)
