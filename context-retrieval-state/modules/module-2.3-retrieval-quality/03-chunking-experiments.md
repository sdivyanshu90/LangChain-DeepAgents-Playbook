# 03 — Chunking Experiments

> **Previous:** [02 → Advanced Retrievers](02-advanced-retrievers.md) | **Next:** [04 → LLM-as-Judge Evaluation](04-llm-as-judge-evaluation.md)

---

## Real-World Analogy

A tailor chooses stitch length based on the fabric.
Fine silk needs small stitches — large ones tear it.
Thick canvas needs large stitches — small ones are wasted effort.

Chunk size is the stitch length of your RAG system.
The right size depends on your document type and your query type.
You find it by experimenting, not by guessing.

---

## Why Experiment Instead of Using Defaults?

Module 2.2 gave you a default: `chunk_size=512, chunk_overlap=50`.
For many cases, that's fine. But retrieval quality degrades in predictable ways
depending on document structure:

```
Technical API docs: short paragraphs per endpoint → chunk_size=256 works well
Legal contracts: dense paragraphs with cross-references → chunk_size=1024 better
Mixed-content reports: varies by section → semantic chunking best
Tabular data (CSV): row-level semantics → chunk_size = one row

No single default works for all document types.
The only way to know is to measure.
```

---

## The Experiment Framework

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import NamedTuple
import json

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ChunkConfig(NamedTuple):
    """Parameters for one chunking experiment."""
    chunk_size: int
    chunk_overlap: int
    label: str

# Define the configurations to test
CONFIGS = [
    ChunkConfig(chunk_size=256, chunk_overlap=25,  label="small_256"),
    ChunkConfig(chunk_size=512, chunk_overlap=50,  label="medium_512"),
    ChunkConfig(chunk_size=1024, chunk_overlap=100, label="large_1024"),
]

def build_vectorstore_for_config(
    pages: list[Document],
    config: ChunkConfig,
) -> FAISS:
    """Build a FAISS index with the given chunking configuration."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    print(f"  [{config.label}] Created {len(chunks)} chunks")
    return FAISS.from_documents(chunks, embeddings)
```

---

## Running the Chunk Size Comparison

```python
# Test questions with known expected answers
TEST_QUERIES = [
    {
        "question": "What is the overtime pay rate for non-exempt employees?",
        "expected_keywords": ["1.5x", "40 hours", "non-exempt"],
    },
    {
        "question": "How many vacation days do employees receive after year one?",
        "expected_keywords": ["15 days", "vacation", "first year"],
    },
    {
        "question": "What is the remote work policy for engineering staff?",
        "expected_keywords": ["remote", "engineering", "days per week"],
    },
]

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Answer ONLY using the context below. "
        "If the answer is not in the context, say 'NOT FOUND'.\n\n"
        "CONTEXT:\n{context}"
    )),
    ("human", "{input}"),
])

def run_experiment(
    pages: list[Document],
    test_queries: list[dict],
) -> dict:
    """
    Run all three chunk configurations against all test queries.
    Returns a dict mapping config_label → list of retrieval results.
    """
    results = {}

    for config in CONFIGS:
        print(f"\nTesting config: {config.label}")
        vectorstore = build_vectorstore_for_config(pages, config)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        doc_chain = create_stuff_documents_chain(llm, GENERATION_PROMPT)
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        config_results = []
        for query_info in test_queries:
            question = query_info["question"]
            expected = query_info["expected_keywords"]

            rag_result = rag_chain.invoke({"input": question})
            answer = rag_result["answer"]
            context_chunks = rag_result["context"]

            # Simple keyword-based relevance check
            keywords_found = [kw for kw in expected if kw.lower() in answer.lower()]
            relevance_score = len(keywords_found) / len(expected)

            config_results.append({
                "question":       question,
                "answer":         answer[:200],
                "keywords_found": keywords_found,
                "relevance":      relevance_score,
                "chunks_used":    len(context_chunks),
                "avg_chunk_len":  sum(len(d.page_content) for d in context_chunks) / max(len(context_chunks), 1),
            })

        results[config.label] = config_results

    return results

# Run and display
loader = PyPDFLoader("./documents/employee-handbook.pdf")
pages = loader.load()

experiment_results = run_experiment(pages, TEST_QUERIES)

# Print comparison table
print("\n" + "="*60)
print("CHUNK SIZE EXPERIMENT RESULTS")
print("="*60)

for config_label, query_results in experiment_results.items():
    avg_relevance = sum(r["relevance"] for r in query_results) / len(query_results)
    avg_chunk_len = sum(r["avg_chunk_len"] for r in query_results) / len(query_results)
    print(f"\n{config_label}:")
    print(f"  Average relevance score: {avg_relevance:.2f}")
    print(f"  Average retrieved chunk length: {avg_chunk_len:.0f} chars")
    for r in query_results:
        print(f"  Q: {r['question'][:50]}...")
        print(f"     Relevance: {r['relevance']:.0%} | Keywords found: {r['keywords_found']}")
```

---

## Expected Results: What the Data Shows

Across typical HR document chunking experiments:

```
Config: small_256
  Average relevance:     0.62
  Chunk length (avg):    ~240 chars
  Observation: Precise matches but context is often cut. Rate is found
               but surrounding eligibility criteria are in different chunks.
               Answers are often incomplete: "1.5x" without the 40-hour threshold.

Config: medium_512  ← sweet spot
  Average relevance:     0.84
  Chunk length (avg):    ~490 chars
  Observation: Good balance. Usually retrieves the relevant section with
               enough surrounding context. The most common correct answers.

Config: large_1024
  Average relevance:     0.71
  Chunk length (avg):    ~980 chars
  Observation: Sometimes retrieves a chunk that contains the answer AND two
               unrelated sections. Model gets confused; gives verbose or hedged answers.
               Higher token cost per query (4 × 1024 chars = 4096 chars of context).
```

---

## The Overlap Experiment

Test different overlap values while holding chunk_size constant:

```python
OVERLAP_CONFIGS = [
    ChunkConfig(chunk_size=512, chunk_overlap=0,   label="no_overlap"),
    ChunkConfig(chunk_size=512, chunk_overlap=50,  label="10pct_overlap"),
    ChunkConfig(chunk_size=512, chunk_overlap=100, label="20pct_overlap"),
    ChunkConfig(chunk_size=512, chunk_overlap=200, label="40pct_overlap"),
]

# Run the same experiment with OVERLAP_CONFIGS
# Expected finding: 10-20% overlap (50-100 chars) usually performs best.
# Higher overlap: chunks become repetitive; diminishing returns; higher storage cost.
```

The 10-20% rule: overlap should be 10-20% of `chunk_size` for prose documents.

---

## Semantic Chunking Experiment

```python
from langchain_experimental.text_splitter import SemanticChunker

# Semantic chunker: let the embedding model decide where topics change
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

# Build the semantic vectorstore
semantic_chunks = semantic_splitter.split_documents(pages)
print(f"Semantic chunking produced {len(semantic_chunks)} chunks")

# Chunk length distribution for semantic vs fixed-size
semantic_lengths = [len(c.page_content) for c in semantic_chunks]
fixed_lengths = [len(c.page_content) for c in
                 RecursiveCharacterTextSplitter(512, 50).split_documents(pages)]

print(f"Semantic: min={min(semantic_lengths)}, "
      f"max={max(semantic_lengths)}, "
      f"avg={sum(semantic_lengths)/len(semantic_lengths):.0f}")
print(f"Fixed:    min={min(fixed_lengths)}, "
      f"max={max(fixed_lengths)}, "
      f"avg={sum(fixed_lengths)/len(fixed_lengths):.0f}")

# Semantic chunking produces variable-length chunks.
# For most documents, variable chunks outperform fixed-size on boundary queries.
# Cost: 1 embedding call per sentence at index time (significantly more expensive).
```

---

## Interpreting and Acting on Results

After running the experiments, follow this decision tree:

```
medium_512 best (most common) → use as-is; done.

small_256 best:
  → Document has very short, dense paragraphs (API docs, specs)
  → Use small chunks; consider ParentDocumentRetriever for context

large_1024 best:
  → Document has very long sections with distributed information
  → Consider using large chunks with ContextualCompression

semantic best:
  → Document has abrupt topic shifts (multi-section reports)
  → Semantic chunking worth the extra cost

All similar:
  → Chunking is not your bottleneck; look at retrieval strategy instead
  → Try MultiQueryRetriever or ParentDocumentRetriever next
```

---

## Common Pitfalls

| Pitfall                                              | What goes wrong                                      | Fix                                                      |
| ---------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| Testing on only 3-5 questions                        | Small sample doesn't represent production traffic    | Use at least 20-30 diverse questions                     |
| Measuring only answer quality, not chunk quality     | You don't know if the right chunks were retrieved    | Log and inspect retrieved chunks per query               |
| Not accounting for cost in the trade-off             | Semantic chunking may perform best but cost 10× more | Include cost per query in your comparison                |
| Fixing chunk size after indexing without re-indexing | Old chunks remain in the vector store                | Re-build the entire index when changing chunk parameters |
| Testing on the same documents used for development   | You've seen these; production has different docs     | Test on a held-out document set                          |

---

## Mini Summary

- No single chunk size works for all document types — experiment to find the optimum.
- `chunk_size=512, chunk_overlap=50` is a reliable default for prose; adjust from there.
- Small chunks: precise retrieval, low context. Large chunks: broad context, low precision.
- Overlap of 10-20% of chunk_size prevents boundary fragmentation.
- Semantic chunking produces higher quality at higher cost — use when fixed-size underperforms.
- Always measure with real queries before concluding which configuration is better.
