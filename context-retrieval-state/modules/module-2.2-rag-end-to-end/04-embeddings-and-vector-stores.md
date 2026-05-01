# 04 — Embeddings and Vector Stores

> **Previous:** [03 → Text Splitters](03-text-splitters.md) | **Next:** [05 → Retrievers and Generation](05-retrievers-and-generation.md)

---

## Real-World Analogy

Imagine mapping every word in a library to a location in a giant building.
Books about "taxation" and "accounting" are stored near each other.
Books about "cooking" are across the building.
"Baking" is closer to "cooking" than "taxation" is.

This is exactly what embedding does — it maps text to a location in a high-dimensional space
where meaning determines proximity.
When you search, you walk to the location that matches your question
and pick up whatever is nearby.

---

## What Is an Embedding?

An embedding converts a piece of text into a vector of floating-point numbers.
The critical property: **similar meaning → similar direction in vector space**.

```
text-embedding-3-small produces 1536-dimensional vectors.

"The dog barked loudly"  → [0.12, -0.07, 0.85, 0.03, ... (1536 values)]
"The canine made noise"  → [0.11, -0.06, 0.87, 0.02, ... (1536 values)]
"Quarterly revenue grew" → [-0.43, 0.72, -0.15, 0.91, ...]

cosine_similarity("dog barked", "canine made noise") ≈ 0.94  ← very similar
cosine_similarity("dog barked", "quarterly revenue") ≈ 0.12  ← unrelated
```

---

## Cosine Similarity — The Measurement

Similarity between two vectors is measured by the angle between them:

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

The result ranges from -1 (opposite) to 1 (identical direction):

```
    Similar meaning:     [A] and [B] point in almost the same direction
                          cos(θ) ≈ 1.0

    Unrelated meaning:   [A] and [B] point at 90°
                          cos(θ) ≈ 0.0

    Opposite meaning:    [A] and [B] point in opposite directions
                          cos(θ) ≈ -1.0 (rare for natural language)
```

When you search a vector store, you embed your query,
then find the `k` stored vectors with the highest cosine similarity.

---

## OpenAIEmbeddings

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()   # loads OPENAI_API_KEY

# Standard production choice
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # text-embedding-3-small: 1536 dimensions, $0.02/1M tokens — best price/quality
    # text-embedding-3-large: 3072 dimensions, $0.13/1M tokens — higher quality
    # text-embedding-ada-002: legacy, 1536 dims — avoid for new projects
)

# Embed a single string (for testing)
vector = embeddings.embed_query("What is the overtime pay rate?")
print(f"Vector length: {len(vector)}")  # 1536
print(f"First 5 values: {vector[:5]}")
# [0.023, -0.041, 0.087, -0.012, 0.064]

# Embed multiple documents at once (batched — more efficient)
doc_vectors = embeddings.embed_documents([
    "Overtime is paid at 1.5x the regular rate.",
    "All employees must complete onboarding within 30 days.",
    "The performance review cycle runs annually in December.",
])
print(f"Embedded {len(doc_vectors)} documents")
print(f"Each vector: {len(doc_vectors[0])} dimensions")
```

---

## Vector Store 1: FAISS (In-Memory)

FAISS is a fast similarity search library from Meta.
It stores vectors in RAM. Fast for development; lost on process restart.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prepare documents
loader = PyPDFLoader("./documents/employee-handbook.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Build the FAISS index from documents
# This makes one embedding API call per chunk (batched internally)
vectorstore = FAISS.from_documents(
    documents=chunks,       # list of Document objects
    embedding=embeddings,   # embedding model to use
)
# ↑ Creates an in-memory index. All chunk vectors are computed and stored.

# Save to disk (so you don't re-embed on every restart)
vectorstore.save_local("./faiss_index")
print("Index saved to ./faiss_index/")

# Load from disk (fast — no embedding calls)
vectorstore_loaded = FAISS.load_local(
    folder_path="./faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,   # required — FAISS uses pickle
)
print("Index loaded from disk")

# Quick similarity search test
results = vectorstore_loaded.similarity_search(
    query="What is the overtime pay rate?",
    k=3,   # return the 3 most similar chunks
)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")
    print(doc.page_content[:200])
```

**FAISS trade-offs:**

| Dimension     | Value                                             |
| ------------- | ------------------------------------------------- |
| Setup         | No server needed; pure Python + C++ library       |
| Persistence   | Manual save/load via `save_local`/`load_local`    |
| Multi-process | No — each process loads its own copy              |
| Scale         | Good to ~1M vectors                               |
| When to use   | Prototyping; offline scripts; single-process apps |

---

## Vector Store 2: Chroma (Persistent Local)

Chroma is a local vector database with built-in persistence.
No save/load calls needed — writes to disk automatically.

```python
from langchain_community.vectorstores import Chroma

# Build index with persistence (writes to disk automatically)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",   # SQLite + binary files written here
    collection_name="employee_handbook",   # logical namespace within Chroma
)
# No save() needed — Chroma writes to persist_directory automatically.

# Load existing index (no re-embedding)
vectorstore_existing = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="employee_handbook",
)

# Add more documents to an existing index (incremental indexing)
new_chunks = splitter.split_documents(new_pages)
vectorstore_existing.add_documents(new_chunks)
# ↑ Embeds and adds new_chunks to the existing collection.

# Delete documents by source
vectorstore_existing._collection.delete(
    where={"source": "old-policy.pdf"}
)
# ↑ Remove all chunks from a specific source (for document updates)
```

**Chroma trade-offs:**

| Dimension     | Value                                                   |
| ------------- | ------------------------------------------------------- |
| Setup         | No server; installs as a Python package                 |
| Persistence   | Automatic — survives restarts                           |
| Multi-process | Limited — SQLite locks; use in single-writer mode       |
| Scale         | Good to ~500K vectors                                   |
| When to use   | Development with persistence; single-server deployments |

---

## Vector Store 3: Pinecone (Cloud)

Pinecone is a fully managed cloud vector database.
Best for production at scale with multiple services.

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os

# Initialise Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index (once, not on every run)
INDEX_NAME = "employee-handbook"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,             # must match embedding model dimensions
        metric="cosine",            # cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Build/connect to the vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    # namespace: logical partition within the index (useful for multi-tenant)
    # namespace="tenant-acme",
)

# Connect to existing index (no re-embedding)
vectorstore_existing = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)
```

**Pinecone trade-offs:**

| Dimension     | Value                                                  |
| ------------- | ------------------------------------------------------ |
| Setup         | Requires Pinecone account; API key from environment    |
| Persistence   | Fully managed; always available                        |
| Multi-process | Yes — designed for distributed access                  |
| Scale         | Billions of vectors                                    |
| When to use   | Production microservices; multi-region; shared indexes |

---

## Choosing a Vector Store

```
Are you prototyping or running tests?
  → FAISS. No setup, no server, no cost.

Do you need persistence without a server?
  → Chroma. Automatic disk persistence; easy to set up.

Do you have multiple services or > 500K vectors?
  → Pinecone. Managed, scalable, no ops work.

Are you in an enterprise that already uses PostgreSQL?
  → pgvector (langchain_postgres.PGVector). No new infrastructure.
```

---

## Common Pitfalls

| Pitfall                                            | What goes wrong                                                                      | Fix                                                                      |
| -------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Mismatched embedding dimensions                    | FAISS or Pinecone rejects the vector                                                 | Use the same embedding model for indexing and querying                   |
| Re-embedding on every startup                      | Slow startup; wasted API cost                                                        | Save FAISS index or use Chroma/Pinecone with persistent storage          |
| Embedding in a loop (not batched)                  | Very slow; hits rate limits                                                          | Use `embed_documents()` which batches automatically                      |
| No `allow_dangerous_deserialization` in FAISS load | `ValueError` when loading a saved FAISS index                                        | Pass `allow_dangerous_deserialization=True` (only load your own indexes) |
| Wrong `k` for retrieval                            | Too small: misses relevant context. Too large: irrelevant context dilutes the answer | Start with `k=4`; adjust based on retrieval quality experiments          |

---

## Mini Summary

- Embeddings map text to high-dimensional vectors; similar meaning → similar direction.
- Cosine similarity measures vector proximity: 1.0 = identical direction; 0.0 = unrelated.
- `OpenAIEmbeddings(model="text-embedding-3-small")` is the default production choice.
- FAISS: in-memory, fast, no server — ideal for development.
- Chroma: persistent, no server — ideal for single-process production.
- Pinecone: fully managed, scalable — ideal for multi-service production.
- Use the same embedding model for indexing and querying — never mix models.
