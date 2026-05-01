# 05 — Citations and Grounding

> **Previous:** [04 → LLM-as-Judge Evaluation](04-llm-as-judge-evaluation.md) | **Next:** [Module 2.4 → Stateful Applications](../../module-2.4-stateful-applications/README.md)

---

## Real-World Analogy

A research paper without citations is an opinion.
A research paper with inline citations is a claim you can verify.

Grounded RAG answers are claims you can verify.
Every statement points back to a specific document and page.
Without citations, a user can't distinguish a correct answer from a hallucination.

---

## Why Citations Are Not Optional

```
Without citations:
  User: "Can I take remote leave while travelling internationally?"
  System: "Yes, remote work is permitted internationally up to 30 days per year."
  User: believes it, submits leave request
  HR: "Where did you get that? Our policy only applies domestically."
  Result: user confusion, broken trust in the AI system.

With citations:
  System: "Remote work is permitted domestically up to 30 days per year.
           [employee-handbook.pdf, p.18]
           Note: International remote work is not mentioned in the retrieved policy."
  User: checks p.18, confirms scope, asks HR for international policy.
  Result: accurate action, user trusted the system appropriately.
```

---

## What Grounding Requires

Three components must work together:

```
1. Retrieval must return metadata-rich chunks
   └─ Every chunk must carry: source filename, page number, section title

2. The prompt must instruct the model to cite
   └─ Explicit instruction: "cite [source, page] after every claim"

3. The output schema must capture source references
   └─ Pydantic model with a citations field that maps each claim to a source
```

---

## Step 1: Ensuring Chunks Carry Metadata

Metadata must be attached at load time and preserved through chunking:

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

loader = PyPDFLoader("./documents/employee-handbook.pdf")
pages = loader.load()
# Each page already has metadata: {"source": "...employee-handbook.pdf", "page": 0}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,   # adds metadata["start_index"] = char offset in original page
)
chunks = splitter.split_documents(pages)

# Verify metadata is present
sample_chunk = chunks[10]
print("Source:", sample_chunk.metadata.get("source"))      # .../employee-handbook.pdf
print("Page:  ", sample_chunk.metadata.get("page"))        # 2 (0-indexed)
print("Start: ", sample_chunk.metadata.get("start_index")) # 1024 (char offset)
print("Content:", sample_chunk.page_content[:80])
```

For web sources, enrich metadata explicitly:

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.company.internal/policy/remote-work")
docs = loader.load()

# WebBaseLoader gives {"source": "https://..."} — add a readable title
for doc in docs:
    doc.metadata["title"] = "Remote Work Policy"
    doc.metadata["retrieved_at"] = "2024-01-15"
```

---

## Step 2: The Citation-Forcing Prompt

A prompt that does NOT explicitly require citations gets answers without them.
The instruction must be specific and the format must be unambiguous:

```python
from langchain_core.prompts import ChatPromptTemplate

CITATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise assistant that answers questions using ONLY the provided context.\n\n"
        "RULES:\n"
        "1. Every factual claim MUST include a citation in the format [source, p.N].\n"
        "2. If a claim cannot be supported by the context, do NOT state it.\n"
        "3. If the context does not contain the answer, say: "
        "'The provided documents do not contain information about this topic.'\n"
        "4. Do NOT infer, extrapolate, or use prior knowledge.\n\n"
        "CONTEXT:\n{context}\n\n"
        "FORMAT EXAMPLE:\n"
        "Non-exempt employees receive overtime at 1.5x their regular rate "
        "[employee-handbook.pdf, p.12]. This applies to all hours beyond "
        "40 per work week [employee-handbook.pdf, p.12].\n"
    )),
    ("human", "{input}"),
])
```

---

## Step 3: Structured Output with a Citations Field

Inline citations in plain text are hard to parse programmatically.
Use a Pydantic schema to extract citations as structured data:

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Citation(BaseModel):
    """One piece of evidence supporting a claim."""
    claim:  str = Field(description="The specific claim being cited")
    source: str = Field(description="Filename of the source document")
    page:   int = Field(description="Page number (1-indexed) of the source")
    quote:  str = Field(description="Exact or near-exact quote from the source that supports the claim")

class GroundedAnswer(BaseModel):
    """Answer with full citation trail."""
    answer:    str             = Field(description="The full answer in plain text with inline citation markers like [1], [2]")
    citations: list[Citation]  = Field(description="List of citations supporting claims in the answer, numbered to match markers")
    confidence: str            = Field(
        description="Confidence level: 'high' (answer fully supported), 'partial' (answer partially supported), 'not_found' (context lacks the answer)"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)
# ↑ Use a strong model for citation-aware generation.
# gpt-4o-mini tends to miss citations or conflate sources.

citation_chain = (
    CITATION_PROMPT
    | llm.with_structured_output(GroundedAnswer)
)
```

---

## End-to-End Grounded RAG Pipeline

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def grounded_answer(question: str) -> GroundedAnswer:
    """
    Run full grounded RAG: retrieve → format context → generate with citations.
    Returns a GroundedAnswer with structured citation data.
    """
    docs = retriever.invoke(question)

    # Format context with explicit source labels so the model can cite accurately
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        page   = doc.metadata.get("page", 0) + 1   # convert 0-indexed to 1-indexed
        context_parts.append(
            f"[Document {i+1}] {source}, page {page}:\n{doc.page_content}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    result = citation_chain.invoke({
        "input":   question,
        "context": context_text,
    })
    return result

# Use it
answer = grounded_answer("What is the overtime rate for non-exempt employees?")

print("Answer:", answer.answer)
print(f"Confidence: {answer.confidence}")
print("\nCitations:")
for i, cit in enumerate(answer.citations, 1):
    print(f"  [{i}] {cit.source}, p.{cit.page}")
    print(f"      Claim: {cit.claim}")
    print(f"      Quote: {cit.quote[:100]}")

# Output:
# Answer: Non-exempt employees receive overtime compensation at 1.5x their regular
#         hourly rate [1]. This applies to all hours worked beyond 40 in a single
#         work week [1].
# Confidence: high
#
# Citations:
#   [1] employee-handbook.pdf, p.12
#       Claim: Overtime rate is 1.5x for non-exempt employees
#       Quote: "Overtime is compensated at 1.5x the regular hourly rate for non-exempt..."
```

---

## Handling Missing Citations Gracefully

What to do when the model returns an answer but confidence is `"not_found"` or citations are empty:

```python
def grounded_answer_with_fallback(question: str) -> dict:
    """
    Full grounded RAG with graceful fallback for missing context.
    """
    docs = retriever.invoke(question)
    context_parts = [
        f"[Document {i+1}] {doc.metadata.get('source','?').split('/')[-1]}, "
        f"page {doc.metadata.get('page', 0) + 1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    context_text = "\n\n---\n\n".join(context_parts)

    result = citation_chain.invoke({"input": question, "context": context_text})

    if result.confidence == "not_found" or not result.citations:
        # Sources retrieved but none relevant — report this honestly
        return {
            "answer": (
                "The available documents do not contain information to answer this question. "
                "Please contact HR directly or consult the full policy library."
            ),
            "citations": [],
            "confidence": "not_found",
        }

    if result.confidence == "partial":
        # Some relevant context but incomplete — signal this to the user
        result.answer = (
            result.answer +
            "\n\n*Note: The available context may not fully cover this topic.*"
        )

    return result.model_dump()
```

---

## Common Pitfalls

| Pitfall                                           | What goes wrong                                           | Fix                                                                                |
| ------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| No citation instruction in the prompt             | Model answers from training data; no citations appear     | Add explicit citation rules to the system prompt                                   |
| 0-indexed page numbers in metadata                | Citations say "p.0" for page 1; users confused            | Always convert: `page = doc.metadata["page"] + 1` before presenting                |
| Relying on model to cite source filename verbatim | Model paraphrases or shortens filename; unverifiable      | Pass explicit `[Document N]` labels in context; model cites `[Document 1]` instead |
| Not checking citation count against claim count   | Model makes 5 claims, provides 2 citations                | Add validation: `len(citations) < n_claims → flag for review`                      |
| Using gpt-4o-mini for citation extraction         | Mini models frequently omit or conflate citations         | Use gpt-4o or an explicit extract-then-generate two-step pattern                   |
| Trusting `confidence="high"` without review       | Model may be overconfident; citations can be hallucinated | Spot-check 10% of `confidence="high"` answers in your eval set                     |

---

## Mini Summary

- Every RAG answer that users act on should include citations to specific sources and pages.
- Citations require three things: metadata-rich chunks, a citation-forcing prompt, and a structured output schema.
- Attach source and page metadata at load time; use `add_start_index=True` for char-level offsets.
- Format context with explicit `[Document N]` labels so the model can cite precisely.
- Use `confidence` field to distinguish complete, partial, and missing-context answers.
- Validate citation count against claim count; flag low-citation answers for review.
