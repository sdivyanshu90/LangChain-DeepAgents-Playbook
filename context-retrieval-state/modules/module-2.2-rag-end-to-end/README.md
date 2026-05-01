# Module 2.2 — RAG End-to-End

> **Track:** Context, Retrieval & State | **Prerequisite:** Module 2.1 Memory Fundamentals

---

## The Full RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline (6 Stages)                               │
│                                                                                 │
│  Stage 1        Stage 2        Stage 3       Stage 4       Stage 5     Stage 6 │
│                                                                                 │
│  ┌────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐   ┌────────┐  ┌──────┐ │
│  │ Load   │──►│  Split   │──►│ Embed  │──►│  Store  │──►│Retrieve│─►│ Gen  │ │
│  │        │   │          │   │        │   │  (vec   │   │        │  │      │ │
│  │ PDF    │   │ chunks   │   │ each   │   │  store) │   │top-k   │  │ LLM  │ │
│  │ Web    │   │ ~512 tok │   │ chunk  │   │         │   │chunks  │  │ +    │ │
│  │ CSV    │   │ overlap  │   │→ float │   │ FAISS / │   │by cos  │  │chunks│ │
│  │ Dir    │   │ = 50 tok │   │  vec   │   │ Chroma  │   │similar │  │→ans  │ │
│  └────────┘   └──────────┘   └────────┘   └─────────┘   └────────┘  └──────┘ │
│                                                                                 │
│  (once at index time)  ──────────────────────────────────┘                     │
│                                                   (at query time) ──────────── │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Topics

| #   | File                                                                     | What you will learn                                         |
| --- | ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| 01  | [01-why-rag-exists.md](01-why-rag-exists.md)                             | Knowledge cutoff, hallucination, retrieval as grounding     |
| 02  | [02-document-loaders.md](02-document-loaders.md)                         | PDF, Web, CSV, Directory loaders; metadata propagation      |
| 03  | [03-text-splitters.md](03-text-splitters.md)                             | Chunking mechanics; size vs overlap math; semantic chunking |
| 04  | [04-embeddings-and-vector-stores.md](04-embeddings-and-vector-stores.md) | Embeddings as geometry; FAISS, Chroma, Pinecone             |
| 05  | [05-retrievers-and-generation.md](05-retrievers-and-generation.md)       | Retrieval chain assembly; MMR; generation with citations    |

---

## Key Packages

```bash
pip install langchain-core langchain-openai langchain-community
pip install pypdf                     # PDF loading
pip install faiss-cpu                 # local vector store
pip install chromadb                  # persistent local vector store
pip install beautifulsoup4 lxml       # web loading
```

---

## How to Work Through This Module

1. **Topic 01** sets the motivation — understand _why_ before building.
2. **Topics 02-04** build the indexing pipeline (run once per document set).
3. **Topic 05** builds the query pipeline (runs on every user question).
4. The `examples/` directory has a working end-to-end script combining all stages.

The model still writes the answer, but the application is responsible for finding the right evidence first.

## Why RAG Exists

LLMs are strong pattern generators, but they are not trustworthy knowledge stores for your private or changing information.

RAG exists because real applications need answers grounded in specific source material:

- internal documentation
- PDFs and policies
- meeting notes
- product specs
- knowledge bases

Without retrieval, the model either guesses or relies on stale training data.

## The End-to-End Pipeline

At a high level, a RAG application does six things:

1. Load source documents.
2. Split them into manageable chunks.
3. Turn those chunks into embeddings.
4. Store the embedded chunks in a vector store.
5. Retrieve the most relevant chunks for a question.
6. Ask the model to answer using only that retrieved context.

That may sound like a lot of moving parts, but each one solves a distinct problem.

## Why Each Piece Matters

### 1. Loaders

Loaders convert raw files or systems into `Document` objects.

Why it matters:

- source data becomes standardized early
- metadata can travel with the content

### 2. Splitters

Splitters break large documents into smaller chunks that fit retrieval and model context windows better.

Why it matters:

- whole documents are often too large and too noisy
- chunking controls retrieval granularity

### 3. Embeddings

Embeddings turn text into vectors so semantically similar content can be found later.

Why it matters:

- retrieval becomes similarity-based rather than keyword-only

### 4. Vector Stores

Vector stores keep the embeddings and the original chunks together.

Why it matters:

- retrieval becomes fast and reusable

### 5. Retrievers

Retrievers define how relevant chunks are selected for a given question.

Why it matters:

- the model only sees what the retriever hands it
- weak retrieval usually leads to weak answers

### 6. Answer Generation

The generation prompt tells the model how to use the retrieved context.

Why it matters:

- grounded context is not enough on its own
- the model still needs instructions about citing, abstaining, and staying within evidence bounds

## Example

See [examples/basic_rag_pipeline.py](examples/basic_rag_pipeline.py).

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
context_docs = retriever.invoke(question)
answer = chain.invoke({"question": question, "context": format_docs(context_docs)})
```

That is the heart of RAG: retrieve first, answer second.

## Best Practices

- preserve useful metadata such as source path and page number
- keep chunk size intentional, not arbitrary
- instruct the model to stay within retrieved evidence
- return citations whenever the application needs traceability
- debug retrieval separately from answer generation

## Common Pitfalls

- treating retrieval as a single solved step
- using chunk sizes that are too large to stay specific
- omitting metadata that later makes citations impossible
- blaming the model when the retriever brought the wrong evidence

## Mini Summary

RAG is not a single trick. It is a pipeline that turns external information into grounded model context.

If the retrieval layer is weak, the answer layer will usually be weak too.

## Optional Challenge

Modify the example so the answer prints the source ids of the retrieved chunks, then compare whether the evidence looks strong enough before trusting the answer.
