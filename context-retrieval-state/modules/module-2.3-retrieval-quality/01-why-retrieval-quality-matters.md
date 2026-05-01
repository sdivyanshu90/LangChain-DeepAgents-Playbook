# 01 — Why Retrieval Quality Matters

> **Previous:** [README → Module Index](README.md) | **Next:** [02 → Advanced Retrievers](02-advanced-retrievers.md)

---

## Real-World Analogy

A surgeon who operates based on the wrong patient's X-ray may perform the surgery
flawlessly — and cause serious harm. The skill is there. The input is wrong.

A language model answering based on the wrong retrieved chunks is in the same position.
The model synthesises fluently and confidently.
But the answer is grounded in the wrong evidence.

---

## The Silent Failure Mode

The most dangerous property of RAG failure is that it looks like success.

```
Query: "What is our return policy for electronics purchased online?"

Scenario A: Correct retrieval
  Retrieved: "Section 4.2: Electronics purchased online may be returned within 30 days
              of delivery. Items must be unopened and in original packaging."
  Answer: "Online electronics may be returned within 30 days in original packaging."
  → Accurate. Grounded. User is correctly informed.

Scenario B: Wrong retrieval (silent failure)
  Retrieved: "Section 4.3: In-store electronics returns are accepted within 14 days
              with receipt. Final sale items are not eligible."
  Answer: "You can return electronics within 14 days. Final sale items are excluded."
  → WRONG. The model has no way to know it retrieved the wrong section.
  → The answer is fluent, confident, and incorrect.
  → The user may act on it. A return may be incorrectly refused.
```

The model's tone, grammar, and structure are identical in both cases.
Only the answer content differs.
**No exception is raised. No error is logged. No flag is set.**

---

## The Garbage In, Garbage Out Principle for RAG

The LLM can only work with what retrieval provides.
It cannot:

- know that a retrieved chunk is irrelevant
- search for additional context you didn't provide
- verify that the retrieval was correct
- refuse to answer based on low-quality context (without explicit instruction)

```
Garbage retrieval → garbage answer, stated confidently.
Good retrieval → good answer, stated confidently.

The model's output quality is bounded by retrieval quality.
A better model with bad retrieval still fails.
The same model with better retrieval produces better answers.
```

This means retrieval quality is the highest-leverage improvement in a RAG system.

---

## Measuring the Failure Rate

Without evaluation, you don't know if your retrieval is working.
A naive RAG system "works" on your test questions because you hand-picked them.
Production traffic is different.

```
Without evaluation:
  You test 10 questions → all work → you ship.
  Production: 1000 questions/day
              → 15% get wrong chunks
              → 15% of answers are confidently wrong
              → discovered 3 weeks later through user complaints

With evaluation (Topic 04):
  You test 50 questions with known answers
  → retrieval precision: 72%
  → you identify which query types fail
  → you fix chunking + add MultiQueryRetriever
  → retrieval precision: 91%
  → you ship with confidence
```

---

## The Four Root Causes of Retrieval Failure

Understanding why retrieval fails lets you fix the right thing:

```
Root Cause 1: Wrong chunk boundaries
─────────────────────────────────────
  The relevant sentence straddles two chunks.
  Neither chunk contains the complete answer.
  Fix: adjust chunk size, increase overlap, use semantic chunking.

Root Cause 2: Query-document vocabulary mismatch
──────────────────────────────────────────────────
  User asks: "How do I reset my password?"
  Document says: "Account credential recovery procedure:"
  Embeddings are not similar because different words are used.
  Fix: MultiQueryRetriever rephrases the query multiple ways.

Root Cause 3: Correct topic, wrong scope
──────────────────────────────────────────
  User asks about "online returns".
  Top chunk discusses "in-store returns" — same topic, wrong scope.
  Fix: ContextualCompressionRetriever re-ranks and filters.

Root Cause 4: Small chunk lacks necessary context
──────────────────────────────────────────────────
  User asks about overtime rate.
  Retrieved chunk: "This applies to non-exempt hourly employees."
  (The rate is in the parent section, not the small chunk.)
  Fix: ParentDocumentRetriever retrieves small chunks but returns parent sections.
```

Each root cause has a specific fix covered in Topic 02.

---

## A Concrete Failure Demonstration

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Create a small document corpus that demonstrates retrieval failure
docs = [
    Document(
        page_content=(
            "Section 4.2: Online Electronics Returns\n"
            "Electronics purchased through our website may be returned within 30 days "
            "of the confirmed delivery date. Items must be in original, unopened packaging. "
            "A prepaid return label will be emailed within 24 hours of submitting the return."
        ),
        metadata={"source": "return-policy.pdf", "page": 4, "section": "4.2"},
    ),
    Document(
        page_content=(
            "Section 4.3: In-Store Electronics Returns\n"
            "Electronics purchased at physical store locations may be returned within 14 days "
            "with original receipt. Final sale and open-box items are non-returnable."
        ),
        metadata={"source": "return-policy.pdf", "page": 4, "section": "4.3"},
    ),
]

# Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Build a small vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})   # k=1 to force failure

# This query should retrieve section 4.2, but may retrieve 4.3 due to chunking
results = retriever.invoke("Can I return an online electronics purchase?")
for doc in results:
    print(f"Section: {doc.metadata.get('section')}")
    print(doc.page_content[:100])
    print("---")
# If section 4.3 is retrieved, the answer will be wrong ("14 days" instead of "30 days").
# The model will state it confidently either way.
```

---

## The Cost of Ignoring Retrieval Quality

```
Impact on user trust:
  First incorrect answer → user notices → trust drops 20%
  Third incorrect answer → user stops relying on the system
  Reputation cost: "The AI gives wrong answers" spreads quickly

Operational cost:
  Customer support escalations from wrong answers
  Legal exposure if the system is used for policy compliance
  Re-work: embedding and indexing cost to rebuild the index

Opportunity cost:
  A working RAG system could deflect 40% of support tickets.
  A broken one deflects 40% incorrectly — worse than no system.
```

---

## Common Pitfalls

| Pitfall                                       | What goes wrong                                                  | Fix                                                   |
| --------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------- |
| Testing only with hand-crafted questions      | You design questions you know will work; production is different | Use a diverse eval set (Topic 04); include hard cases |
| Assuming higher-quality model fixes retrieval | The model can't recover from wrong context                       | Fix retrieval first; upgrade model last               |
| Single chunk per query (k=1)                  | Any retrieval error returns wrong answer                         | Use k=4 minimum; multiple chunks provide redundancy   |
| Not logging retrieved chunks in production    | You can't debug wrong answers without knowing what was retrieved | Always log `result["context"]` for every query        |
| Shipping without an eval baseline             | You can't measure improvement                                    | Build an eval dataset before optimising               |

---

## Mini Summary

- Silent failure is the primary danger: the model answers confidently from wrong chunks.
- The model cannot know if its context is correct — retrieval quality is your responsibility.
- The four root causes: wrong boundaries, vocabulary mismatch, wrong scope, missing parent context.
- Each root cause has a specific retriever fix covered in Topic 02.
- Build an evaluation dataset and measure retrieval precision before shipping.
- Retrieval quality is the highest-leverage improvement in a RAG system.
