# 01 — Why RAG Exists

> **Previous:** [README → Module Index](README.md) | **Next:** [02 → Document Loaders](02-document-loaders.md)

---

## Real-World Analogy

A brilliant doctor who graduated in 2022 knows a great deal about medicine.
But if you ask them about a drug approved in 2024, they genuinely don't know.
They're not lying — they simply weren't trained on that information.

Now put that doctor in a room with the drug's clinical trial report.
They can read it and give you an accurate, informed answer.

RAG is the process of putting the relevant document in the room.

---

## The Knowledge Cutoff Problem

Every LLM is trained on data up to a certain date.
After that date, the model has no awareness of:

- New regulations or policy changes
- Your company's internal documentation
- Product updates after the training cutoff
- Customer contracts, case files, or tickets
- Any information that was never publicly available online

```
Training data → frozen knowledge at cutoff date

  GPT-4o training cutoff: ~April 2024
  Today: April 2026

  "What are the new EU AI Act compliance requirements for high-risk AI?"
  → Model will either confuse with earlier drafts, or say "I don't know."
  → RAG fetches the actual regulation text and grounds the answer.

  "What does our employee handbook say about remote work?"
  → Model has never seen your handbook.
  → RAG indexes it and retrieves the relevant section.
```

---

## The Hallucination Problem

A model asked about something outside its training data does not say "I don't know"
by default — it generates a plausible-sounding answer.

```
User: "What does Section 4.2 of our software license say about sublicensing?"

Without RAG (model guessing):
  "Section 4.2 typically covers sublicensing rights and states that the licensee
   may sublicense to affiliates subject to written consent..."
  ↑ COMPLETELY FABRICATED — the model has never seen this license.
  ↑ Sounds authoritative. May be wrong. Could be legally damaging.

With RAG (model grounded on retrieved text):
  Retrieved chunk: "4.2 Sublicensing: No sublicensing is permitted without prior
  written consent from the Licensor. Affiliates are not exempt from this requirement."

  Model response: "According to Section 4.2 of your license agreement, sublicensing
  is not permitted without prior written consent. This explicitly includes affiliates."
  ↑ Grounded in the actual text. Accurate and citable.
```

This is the reliability equation RAG changes:

- Without RAG: model response quality = training data coverage × prompt quality.
- With RAG: model response quality = retrieved document quality × prompt quality.

You now control the quality of the input. You cannot control what the model was trained on.

---

## Why Retrieval Changes the Equation

Retrieval separates the _what to say_ from the _how to say it_.

```
Traditional LLM:
  ┌──────────────────────────────────┐
  │  Everything the model knows      │
  │  (frozen at training cutoff)     │
  │                                  │
  │  "How does our API work?"   ──►  │ "I don't have information about your API."
  │                                  │  or hallucinates plausibly
  └──────────────────────────────────┘

RAG-augmented LLM:
  ┌──────────────────────────────────┐
  │  Retrieved documents (dynamic)   │
  │  fetched at query time           │
  │                                  │
  │  [API docs chunk 1]              │
  │  [API docs chunk 2]         ──►  │ "According to your API documentation,
  │  [API docs chunk 3]              │  the /users endpoint accepts..."
  └──────────────────────────────────┘
         ↑ from your knowledge base
```

The model's job is now synthesis and explanation, not recall.
Synthesis is what language models are actually good at.

---

## The Two Phases of RAG

Understanding the two distinct phases prevents architecture mistakes:

```
Phase 1: Indexing (run once, or when documents change)
──────────────────────────────────────────────────────
  Load → Split → Embed → Store in vector database

  Time: minutes to hours (depends on document count)
  Cost: embedding API calls (once per document chunk)
  When: when documents are added, updated, or deleted
  Result: a searchable vector index

Phase 2: Querying (run on every user question)
───────────────────────────────────────────────
  Embed question → Search vector store → Retrieve top-k → Generate answer

  Time: <2 seconds typically
  Cost: 1 embedding call + 1 LLM call per question
  When: every time a user asks a question
  Result: a grounded answer with source references
```

Many bugs in RAG systems come from running Phase 1 logic in Phase 2 code, or vice versa.
Keep them separate in your codebase.

---

## What RAG Does NOT Fix

RAG is not a universal solution.
Understanding its limits prevents misapplication:

```
RAG does NOT fix:
  ✗  Poor document quality — garbage in, garbage out
  ✗  Missing documents — retrieval can only find what's been indexed
  ✗  Retrieval failures — if the wrong chunks come back, the answer is wrong
  ✗  Conflicting documents — if two sources contradict, the model may pick either
  ✗  Numerical reasoning — the model still reasons; just with better input data
  ✗  Very long context requirements — top-k retrieval picks only a few chunks

RAG DOES fix:
  ✓  Knowledge cutoff — any document indexed after training is accessible
  ✓  Private data — documents never publicly available are now accessible
  ✓  Citation and traceability — you know exactly which document was used
  ✓  Hallucination on factual questions — grounded in retrieved text
  ✓  Keeping answers up to date — re-index when documents change
```

---

## RAG vs Fine-Tuning: When to Choose RAG

This is a common architectural decision point:

| Concern                                      | RAG                              | Fine-Tuning                      |
| -------------------------------------------- | -------------------------------- | -------------------------------- |
| Documents change frequently                  | ✓ Re-index; no retraining        | ✗ Retrain expensive              |
| Large private knowledge base (1000s of docs) | ✓ Efficient at scale             | ✗ Context limits                 |
| Need citations and source tracing            | ✓ Built-in                       | ✗ Model can't cite training data |
| Need model to reason in a new domain style   | ✗ RAG doesn't change reasoning   | ✓ Fine-tuning shapes style       |
| Cost                                         | ✓ Lower (retrieval + generation) | ✗ High (GPU training)            |
| Time to deploy                               | ✓ Hours                          | ✗ Days to weeks                  |

**Rule of thumb:** If your problem is "the model doesn't know about X", use RAG.
If your problem is "the model doesn't behave like Y", use fine-tuning.

---

## Common Pitfalls

| Pitfall                                      | What goes wrong                                        | Fix                                                        |
| -------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| Indexing everything in one pass              | Large documents take hours; no progress visibility     | Process in batches with progress logging                   |
| Forgetting to re-index when documents change | Stale answers; model cites outdated content            | Build a document change detection and re-indexing pipeline |
| Using RAG for reasoning tasks                | RAG provides information; it doesn't improve reasoning | Use chain-of-thought prompting for reasoning, not RAG      |
| Putting the whole document in context        | Token limit exceeded; cost explosion                   | Always chunk documents (Topic 03)                          |
| Not tracking which document was retrieved    | Cannot cite sources or debug wrong answers             | Always store metadata with chunks (Topic 02)               |

---

## Mini Summary

- LLMs have a training cutoff; they cannot access private, new, or proprietary data.
- Without grounding, models hallucinate plausibly — the most dangerous failure mode.
- RAG separates the problem: retrieval finds relevant facts, the model synthesises them.
- Two phases: indexing (once, offline) and querying (per request, online).
- RAG fixes knowledge gaps; it does not fix bad documents or poor retrieval.
- Choose RAG when the knowledge changes; choose fine-tuning when the behaviour needs to change.
