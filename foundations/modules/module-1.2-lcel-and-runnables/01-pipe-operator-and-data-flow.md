# 01 — The Pipe Operator and Data Flow

> **Previous:** [Module 1.1](../module-1.1-what-is-langchain/README.md) | **Next:** [02 → Runnable Types](02-runnable-types.md)

---

## Real-World Analogy

Unix pipes: `cat file.txt | grep "error" | sort | uniq -c`

Each command does one thing. The output of one becomes the input of the next.
The shell does not execute anything until you press Enter — it builds the pipeline lazily first.

LCEL is the same pattern for LLM workflows.

---

## What `|` Actually Does

```python
chain = prompt | model | parser
```

This calls `prompt.__or__(model)` which creates a `RunnableSequence([prompt, model])`.
Then `RunnableSequence.__or__(parser)` extends it to `RunnableSequence([prompt, model, parser])`.

**No computation happens at definition time.**
The pipeline is a description of the data flow.
Computation starts only when you call `.invoke()`, `.stream()`, or `.batch()`.

```
chain = prompt | model | parser
         ↑ just defines the pipeline

chain.invoke({"question": "..."})
         ↑ NOW it executes
```

### Why Lazy Evaluation Matters

1. **Introspection** — you can inspect the chain before running it: `chain.get_graph().print_ascii()`
2. **Composition** — you can pass a chain as an argument, embed it inside `RunnableParallel`, without triggering execution
3. **Reuse** — the same chain object can be invoked multiple times with different inputs

---

## The Data Flow Contract

Every stage has an explicit input type and output type.
The `|` operator chains stages where the output type of stage N matches the input type of stage N+1.

```
Stage               Input type          Output type
────────────────────────────────────────────────────
ChatPromptTemplate  dict                list[BaseMessage]
ChatOpenAI          list[BaseMessage]   AIMessage
StrOutputParser     AIMessage           str
JsonOutputParser    AIMessage           dict
PydanticOutputParser AIMessage          BaseModel
```

Visualized as a pipeline:

```
{"question": "What is 2+2?"}
        │
        │  dict
        ▼
  ChatPromptTemplate
        │
        │  [SystemMessage("..."), HumanMessage("What is 2+2?")]
        ▼
     ChatOpenAI
        │
        │  AIMessage(content="4")
        ▼
  StrOutputParser
        │
        │  "4"
        ▼
  application code
```

If you accidentally pipe `StrOutputParser` into `ChatOpenAI`, you get a type error:
`ChatOpenAI` expects `list[BaseMessage]`, not `str`. LCEL surfaces this at runtime clearly.

---

## Inspecting a Chain

```python
chain = prompt | model | StrOutputParser()

# Print a text diagram of the pipeline
chain.get_graph().print_ascii()
# PromptTemplate → ChatOpenAI → StrOutputParser

# See the full node structure
for node in chain.get_graph().nodes.values():
    print(node.name, "→", node.data)
```

For LangGraph agents (Level 3+), the equivalent is `graph.get_graph().draw_mermaid_png()`.

---

## The Equivalent Without LCEL

Understanding what `|` replaces makes its value clear:

```python
# ❌ Without LCEL — manual orchestration
def run_chain(question: str) -> str:
    messages = prompt.format_messages(question=question)
    response = model.invoke(messages)
    result = parser.invoke(response)
    return result

# ✅ With LCEL — explicit data flow
chain = prompt | model | parser
result = chain.invoke({"question": question})
```

The LCEL version:

- Is inspectable (`.get_graph()`)
- Supports `.stream()` without changing any component
- Can be embedded inside `RunnableParallel` without refactoring
- Traces automatically in LangSmith (all stages, not just the final call)

---

## Tracing in LangSmith

When LangSmith tracing is enabled, every `|` boundary creates a separate span:

```
LangSmith trace for chain.invoke({"question": "..."})
├── ChatPromptTemplate.invoke     10ms
├── ChatOpenAI.invoke             850ms   ← token count, cost, finish reason
└── StrOutputParser.invoke         1ms
Total: 861ms   Tokens: 94 prompt + 12 completion
```

This is why LCEL is preferred: you get per-step observability for free.

---

## Common Pitfalls

| Pitfall                              | What you see                                | Fix                                                         |
| ------------------------------------ | ------------------------------------------- | ----------------------------------------------------------- |
| Piping incompatible types            | `TypeError` or `ValidationError` at runtime | Check the output type of each stage before piping           |
| Calling the chain at definition time | `RecursionError` or unexpected execution    | `chain = a \| b` (defines); `chain.invoke(...)` (executes)  |
| Mutating inputs inside a stage       | Downstream stages receive modified state    | Treat each stage as a pure function; return new dicts       |
| Deeply nested lambdas inside `\|`    | Unreadable, untestable pipeline             | Extract to named functions; use `RunnableLambda` explicitly |

---

## Mini Summary

- `|` creates a `RunnableSequence` lazily — nothing runs until `.invoke()`.
- Data flows as typed objects between stages: `dict → list[BaseMessage] → AIMessage → str/Model`.
- `.get_graph()` lets you inspect the pipeline structure before running it.
- LCEL gives LangSmith per-step tracing for free — no annotations needed.

---

## Next: [02 → Runnable Types Deep Dive](02-runnable-types.md)
