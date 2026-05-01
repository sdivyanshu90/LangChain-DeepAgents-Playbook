# Module 2.3 — Retrieval Quality

> **Track:** Context, Retrieval & State | **Prerequisite:** Module 2.2 RAG End-to-End

---

## The Central Insight: Retrieval Is the Control Surface

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│         Retrieval quality controls answer quality                    │
│                                                                      │
│   Weak retrieval:                   Strong retrieval:               │
│                                                                      │
│   Question → wrong chunks           Question → right chunks         │
│            → model is confused               → model is grounded    │
│            → confident wrong answer          → accurate answer      │
│                                                                      │
│   The model cannot fix bad retrieval.                               │
│   A better model with bad retrieval still produces bad answers.     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

The model is a synthesiser. It works with what retrieval provides.
Improve the retrieval, and the answers improve — without changing the model.

---

## Topics

| #   | File                                                                       | What you will learn                                                          |
| --- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 01  | [01-why-retrieval-quality-matters.md](01-why-retrieval-quality-matters.md) | Silent failure mode; garbage in, garbage out                                 |
| 02  | [02-advanced-retrievers.md](02-advanced-retrievers.md)                     | MultiQueryRetriever; ContextualCompressionRetriever; ParentDocumentRetriever |
| 03  | [03-chunking-experiments.md](03-chunking-experiments.md)                   | Comparing chunk sizes; overlap experiments; semantic chunking                |
| 04  | [04-llm-as-judge-evaluation.md](04-llm-as-judge-evaluation.md)             | LLM scoring retrieval quality; RAGAS framework; building an eval dataset     |
| 05  | [05-citations-and-grounding.md](05-citations-and-grounding.md)             | Extracting source+page; citation formats; forcing the model to cite          |

---

## The Retrieval Quality Ladder

```
Level 1 (naive):     Simple similarity search → top-k chunks
Level 2 (better):    MMR retrieval → diverse chunks
Level 3 (advanced):  MultiQuery + Compression → expanded + filtered
Level 4 (best):      ParentDocument + Eval loop → size + quality tuned
```

Progress through the ladder as your RAG system matures.

---

## Key Packages

```bash
pip install langchain-core langchain-openai langchain-community ragas
```

---

## How to Work Through This Module

1. Topic 01: understand why weak retrieval is the dominant failure mode.
2. Topic 02: learn the three advanced retrievers and when to use each.
3. Topic 03: run the chunking experiments on your own documents.
4. Topic 04: build an eval dataset and measure your retrieval quality.
5. Topic 05: add citations to every answer.

That is what weak retrieval quality looks like in a RAG system.

## Why Retrieval Quality Deserves Its Own Module

Many beginners build a basic RAG pipeline and assume the hard part is done. It is not.

In production systems, retrieval quality is often the main bottleneck. If the wrong chunks enter the prompt, the model cannot recover reliably.

This is why retrieval quality matters:

- chunking changes what the retriever can see
- overlap changes whether important details survive boundaries
- metadata filters change which documents are even eligible
- reranking can rescue weak top-k results
- citation formatting affects trust and debugging
- evaluation reveals whether retrieval is actually improving

## Key Tuning Levers

### 1. Chunk Size

Small chunks:

- more specific
- easier to retrieve precisely

Risk:

- can lose surrounding context

Large chunks:

- preserve more context

Risk:

- reduce precision and increase noise

### 2. Chunk Overlap

Overlap helps preserve ideas that cross chunk boundaries.

Risk:

- too much overlap creates redundancy and wasted prompt budget

### 3. Metadata Filters

Filters help restrict retrieval to the right scope, such as a product area, policy type, or date range.

Why it matters:

- relevance often depends on context, not just similarity

### 4. Reranking

Reranking helps reorder initially retrieved results using a stronger relevance pass.

Why it matters:

- raw vector similarity is useful, but not always enough

### 5. Citations and Evaluation

If you cannot inspect the evidence, you cannot debug retrieval confidently.

Good retrieval systems make evidence visible and measurable.

## Example

See [examples/retrieval_diagnostics.py](examples/retrieval_diagnostics.py).

The example compares multiple chunking settings and prints the retrieved chunk metadata for the same question. That is the right debugging mindset: compare the evidence path before changing the prompt.

## Best Practices

- test chunk strategies with representative questions
- keep source metadata rich enough for citations and filters
- inspect retrieved chunks directly before tuning the generation prompt
- evaluate retrieval with targeted question sets, not intuition alone
- add reranking only after establishing a good baseline retriever

## Common Pitfalls

- assuming the top retrieved result is automatically good enough
- tuning the prompt when the retrieval evidence is weak
- treating citations as presentation-only instead of debugging data
- optimizing for a single query instead of a question set

## Mini Summary

Retrieval quality is the control surface of RAG.

If you do not tune chunking, filtering, citations, and evaluation deliberately, the system will appear inconsistent even when the model is behaving reasonably.

## Optional Challenge

Take a small document set and compare two chunk sizes with the same question list. Record which setting produces the strongest evidence traces.
