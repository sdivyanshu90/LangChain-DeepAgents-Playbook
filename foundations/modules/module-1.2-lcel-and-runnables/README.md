# Module 1.2 — LCEL and Runnables

> **DeepAgent Level: 1** | Estimated reading time: 50 min | Prerequisites: Module 1.1

LCEL (LangChain Expression Language) is the composition model that turns individual
primitives into pipelines. It is not syntax sugar — it is a typed, lazy, streaming-native
data-flow graph that replaces ad hoc glue code.

---

## Why LCEL Is the Core Skill

Every LangChain pattern in Level 2 and Level 3 is expressed in LCEL.
RAG chains, agent nodes, structured extractors, parallel runners — they are all
Runnables composed with `|`.

If you understand LCEL, you can read any LangChain codebase.
If you do not, every advanced example will feel like magic.

---

## Topics in This Module

| # | File | What you will learn |
|---|---|---|
| 01 | [The Pipe Operator and Data Flow](01-pipe-operator-and-data-flow.md) | How `\|` works, lazy evaluation, the Runnable contract, type shapes at each stage |
| 02 | [Runnable Types Deep Dive](02-runnable-types.md) | RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch |
| 03 | [Streaming and Batching](03-streaming-and-batching.md) | `.stream()`, `.astream()`, `.batch()`, `.abatch()`, when and why to use each |
| 04 | [Data Shape Discipline](04-data-shape-discipline.md) | Tracing types through a chain, debugging shape mismatches, `RunnablePassthrough.assign()` |
| 05 | [Composition Patterns](05-composition-patterns.md) | Fan-out, fan-in, conditional branching, sequential enrichment, real-world recipes |

---

## The Runnable Contract Visualized

```
Every Runnable object guarantees:

  .invoke(input)              → output            (sync,  one item)
  .stream(input)              → Iterator[chunk]   (sync,  streaming)
  .batch([input1, input2])    → [output1, output2](sync,  parallel)
  .ainvoke(input)             → Awaitable[output] (async, one item)
  .astream(input)             → AsyncIterator     (async, streaming)
  .abatch([...])              → Awaitable[list]   (async, parallel)

Because every component speaks the same interface, any component
can be substituted for any other with the same input/output shape.
```

---

## Start With: [01 → The Pipe Operator and Data Flow](01-pipe-operator-and-data-flow.md)
