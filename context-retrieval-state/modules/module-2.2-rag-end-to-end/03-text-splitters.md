# 03 — Text Splitters

> **Previous:** [02 → Document Loaders](02-document-loaders.md) | **Next:** [04 → Embeddings and Vector Stores](04-embeddings-and-vector-stores.md)

---

## Real-World Analogy

Imagine indexing an encyclopedia by printing each sentence on a separate index card.
If your cards are too small, every answer requires dozens of cards with no context.
If your cards are too large, each card covers unrelated topics that dilute the search.

The right card size preserves enough context to be useful without including so much
that the relevant signal is buried in noise. That is the chunking problem.

---

## Why Chunking Is Necessary

Documents can be thousands of pages long.
You cannot retrieve useful information from a 10,000-word document as a single unit.

```
Problem with no chunking:
  ┌─────────────────────────────────────────────────────────────┐
  │  200-page policy document as one Document                   │
  │                                                             │
  │  User: "What is the overtime pay rate?"                     │
  │                                                             │
  │  Retrieval: returns the entire 200-page document            │
  │  → 150,000 tokens — exceeds context window                  │
  │  → Model can't focus on the specific section                │
  │  → Slow and expensive                                       │
  └─────────────────────────────────────────────────────────────┘

Problem with chunk size = 1 sentence:
  ┌─────────────────────────────────────────────────────────────┐
  │  "Overtime is compensated at 1.5x"                          │
  │  ↑ Too small — missing the policy context around it         │
  │  What overtime? Which employees? When does it apply?        │
  │  The model can't give a useful answer from this alone.      │
  └─────────────────────────────────────────────────────────────┘
```

---

## RecursiveCharacterTextSplitter — The Default Choice

`RecursiveCharacterTextSplitter` is LangChain's recommended splitter.
It tries to split on natural boundaries in priority order:
`["\n\n", "\n", " ", ""]` (paragraphs, then lines, then words, then characters).

```
Splitting strategy:
  1. Try to split on "\n\n" (paragraph breaks) — preserves semantic units
  2. If chunk is still too large, split on "\n" (line breaks)
  3. If still too large, split on " " (word boundaries)
  4. If still too large, split on "" (character by character — last resort)
```

This hierarchy means the splitter almost never cuts mid-word or mid-sentence.

---

## chunk_size and chunk_overlap — The Core Parameters

```
chunk_size:    maximum characters (or tokens) per chunk
chunk_overlap: how many characters the end of one chunk shares with the start of the next

Visual example (chunk_size=20, chunk_overlap=5):

Source text: "The quick brown fox jumps over the lazy dog near the pond"

Chunk 1: "The quick brown fox "   (20 chars)
                           ^────────────┐
Chunk 2:             "fox jumps over t"  (overlap starts at "fox")
                                     ^──────────┐
Chunk 3:                         "the lazy dog"
```

### Why Overlap Exists

Overlap solves the boundary problem: a key sentence might straddle a chunk boundary.

```
Without overlap (chunk_size=100, chunk_overlap=0):

  Chunk 1: "The maximum overtime rate is 1.5x for hours worked beyond 40 per week."
  Chunk 2: "This applies to all hourly employees classified as non-exempt under FLSA."

  Query: "Who gets overtime pay?"
  Retrieval might only return Chunk 2 — which has the answer, but lacks the rate.
  Or only Chunk 1 — which has the rate, but not who it applies to.

With overlap (chunk_size=100, chunk_overlap=50):

  Chunk 1: "The maximum overtime rate is 1.5x for hours worked beyond 40 per week."
  Chunk 2: "1.5x for hours worked beyond 40 per week. This applies to all hourly
            employees classified as non-exempt under FLSA."
  ↑ The rate and the scope are now co-located in Chunk 2.
```

---

## Code: RecursiveCharacterTextSplitter in Practice

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load source documents
loader = PyPDFLoader("./documents/employee-handbook.pdf")
pages = loader.load()   # list of Document objects, one per page

# Create the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # target: ~512 characters per chunk
    chunk_overlap=50,         # 50 characters of overlap between adjacent chunks
    length_function=len,      # measure length in characters (not tokens)
    add_start_index=True,     # adds "start_index" to metadata — useful for debugging
    # separators: default is ["\n\n", "\n", " ", ""]
    # Override for code files: ["\nclass ", "\ndef ", "\n", " ", ""]
)

# Split all pages into chunks
chunks = splitter.split_documents(pages)
# ↑ Returns a new list of Documents.
# Each chunk has the same metadata as its parent page, plus start_index.

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
# Pages: 45
# Chunks: 312 (approximately 7 chunks per page for a 512-char chunk size)

# Inspect a single chunk
print(chunks[5].page_content)
# "...the employee must submit a timesheet by Friday. Overtime rates apply
#  to all hours exceeding the standard 40-hour work week..."
print(chunks[5].metadata)
# {"source": "employee-handbook.pdf", "page": 2, "start_index": 1536}
```

---

## Chunk Size Experiments: The Trade-Off in Numbers

Different chunk sizes produce different retrieval characteristics:

```
Experiment: same question, same document, different chunk sizes.
Document: 50-page technical spec (50,000 characters total)
Question: "What is the API rate limit for the /users endpoint?"

Chunk size 256 (small):
  Chunks created: ~195
  Typical chunk: "rate_limit: 100 requests/minute for /users"
  ✓ High precision — exactly what you need
  ✗ Low recall — surrounding context (auth requirements, error codes) missed

Chunk size 512 (medium):
  Chunks created: ~98
  Typical chunk: "Rate Limits\nAll endpoints enforce rate limits.
    /users: 100 req/min. /orders: 50 req/min. Authentication required..."
  ✓ Good balance — relevant section with context
  ✓ Usually the right default

Chunk size 1024 (large):
  Chunks created: ~49
  Typical chunk: entire API reference section + unrelated configuration section
  ✗ Low precision — answer buried in irrelevant content
  ✓ Good recall — rarely misses context
  ✗ Higher token cost per retrieval
```

**Default recommendation:** `chunk_size=512, chunk_overlap=50`.
Adjust based on your document structure and retrieval experiments (covered in Module 2.3).

---

## The Overlap Math

When you set `chunk_size=512` and `chunk_overlap=50`:

$$\text{effective new content per chunk} = 512 - 50 = 462 \text{ characters}$$

$$\text{chunks from document of length } L = \lceil L / 462 \rceil$$

For a 50,000-character document: $\lceil 50000 / 462 \rceil \approx 109$ chunks.

The overlap adds ~10% storage overhead but significantly improves retrieval quality.

---

## The Pitfall: Cutting Mid-Sentence

`RecursiveCharacterTextSplitter` tries to avoid this, but it can still happen
if a sentence is longer than `chunk_size`.

```python
# Problem: a very long sentence that exceeds chunk_size
text = "According to Article 7, Clause 3 of the Service Level Agreement " \
       "signed on January 15 2024 between AcmeCorp and TechServices Inc, " \
       "the maximum permissible downtime per quarter is four hours, " \
       "excluding planned maintenance windows pre-approved in writing."
# Length: ~280 characters — fits in one 512-char chunk, fine.

# But if chunk_size=100:
small_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
result = small_splitter.create_documents([text])
for chunk in result:
    print(repr(chunk.page_content))
# "According to Article 7, Clause 3 of the Service Level Agreement signed on
#  January 15 2024 between"
# ← Cuts at a word boundary, but the meaning is fragmented.
# The "4 hours" limit is in the next chunk, disconnected from the agreement it belongs to.
```

Use `chunk_size >= 400` to avoid this in typical prose documents.

---

## Semantic Chunking: An Introduction

`RecursiveCharacterTextSplitter` splits on character count.
Semantic chunking splits on meaning — keeping semantically related sentences together.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# SemanticChunker uses embeddings to detect where topic shifts occur
# Expensive: makes one embedding call per sentence
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95,          # split when similarity drops below 95th percentile
)

docs = semantic_splitter.split_text(long_document_text)
# Chunks are semantically coherent but variable in length.
# More expensive (1 embedding per sentence) than character splitting.
# Worth it for long documents with mixed topics.
```

**When to use semantic chunking:**

- Documents that switch topics abruptly (mixed-content PDFs, reports)
- When character-based chunking consistently misses relevant context
- When you can afford the additional embedding cost

**When to stick with RecursiveCharacterTextSplitter:**

- Most cases — it's fast, cheap, and good enough
- When documents have consistent structure (APIs, handbooks, specs)

---

## Common Pitfalls

| Pitfall                                       | What goes wrong                                                      | Fix                                                                                |
| --------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `chunk_size` too small (< 100)                | Context is lost; retrieval finds the right area but too little text  | Use at least 256; 512 is a safe default                                            |
| `chunk_overlap=0`                             | Sentences straddling chunk boundaries never appear in the same chunk | Use 10-20% of `chunk_size` as overlap                                              |
| Using `length_function=len` with token budget | Character counts are not token counts; chunks may exceed token limit | Use a token counter function for token-budget-based chunking                       |
| Splitting code files by character             | Functions and classes get cut arbitrarily                            | Use `Language.PYTHON` separator list for code files                                |
| Not propagating metadata                      | Chunks lose their source information                                 | Always use `split_documents()`, not `split_text()` — the former preserves metadata |

---

## Mini Summary

- Chunking is essential: documents too large to retrieve whole must be split into focused units.
- `RecursiveCharacterTextSplitter` splits on natural text boundaries (paragraphs → lines → words).
- `chunk_size=512, chunk_overlap=50` is a reliable default for prose documents.
- Overlap prevents key sentences from being stranded at chunk boundaries.
- Use `split_documents()` (not `split_text()`) to preserve metadata.
- Semantic chunking is more precise but more expensive — use when structure matters more than speed.
