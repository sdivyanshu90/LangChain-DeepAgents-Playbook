# 05 — Retrievers and Generation

> **Previous:** [04 → Embeddings and Vector Stores](04-embeddings-and-vector-stores.md) | **Next:** [Module 2.3 → Retrieval Quality](../../module-2.3-retrieval-quality/README.md)

---

## Real-World Analogy

A research librarian does two jobs:

1. **Find** the most relevant references for your question.
2. **Synthesise** those references into a coherent answer.

The first job is retrieval. The second is generation.
A RAG chain automates both, keeping the two stages cleanly separated.

---

## The Retrieval Chain Architecture

```
User Question
      │
      ▼
┌─────────────┐    embed question    ┌───────────────┐
│   Retriever │─────────────────────►│  Vector Store │
│ (top-k)     │◄─────────────────────│  similarity   │
└──────┬──────┘    return k chunks   └───────────────┘
       │
       │ [Document, Document, Document, Document]
       ▼
┌─────────────────────────────────────────────────┐
│   Generation Prompt                             │
│                                                 │
│   System: "Answer using only the context below."│
│   Context: {chunk 1 text}\n{chunk 2 text}\n...  │
│   Human:   "{user question}"                    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
                    ┌─────┐
                    │ LLM │
                    └──┬──┘
                       │
                       ▼
                   Answer + Sources
```

---

## `as_retriever()` — Converting a Vector Store to a Retriever

Any vector store can become a `Retriever` with `.as_retriever()`.
A `Retriever` is a Runnable that accepts a string query and returns `list[Document]`.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# Default retriever: similarity search, top 4 results
retriever = vectorstore.as_retriever(
    search_type="similarity",     # cosine similarity search (default)
    search_kwargs={"k": 4},       # return 4 chunks per query
)

# Test the retriever standalone
docs = retriever.invoke("What is the overtime pay rate?")
print(f"Retrieved {len(docs)} documents")
for doc in docs:
    print(f"  → Page {doc.metadata['page']}: {doc.page_content[:80]}")
```

---

## Similarity Search vs MMR

**Similarity Search:** returns the top-k most similar chunks.
Risk: if 4 very similar chunks all say the same thing, you get redundant context.

**MMR (Maximal Marginal Relevance):** balances similarity with diversity.
It selects chunks that are relevant to the query AND different from each other.

```python
# MMR retriever: relevant but diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,             # final number of results to return
        "fetch_k": 20,      # candidates to consider before applying diversity
        "lambda_mult": 0.5, # 0.0 = pure diversity; 1.0 = pure similarity; 0.5 = balanced
    },
)

# When to use MMR vs similarity:
# Similarity:  short, specific factual questions — you want the best match
# MMR:         broad questions — you want diverse coverage, not 4 copies of the same fact
```

---

## The Generation Prompt

The generation prompt is the most important part of the RAG chain.
It must:

1. Instruct the model to use ONLY the provided context (grounding).
2. Tell the model what to say when the context doesn't have the answer.
3. Request source citations.

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant that answers questions based on provided context.\n\n"
        "Rules:\n"
        "1. Answer ONLY using the information in the CONTEXT section below.\n"
        "2. If the context does not contain the answer, say: "
        "'I could not find information about this in the provided documents.'\n"
        "3. Do NOT use knowledge from your training data.\n"
        "4. After your answer, list the sources you used in this format:\n"
        "   Sources: [filename, page N]\n\n"
        "CONTEXT:\n{context}"
    )),
    ("human", "{question}"),
])
```

Rule 2 is critical: it prevents the model from falling back to training data
when the retrieved documents don't have the answer.
Without it, the model will hallucinate confidently rather than admitting uncertainty.

---

## `create_stuff_documents_chain` — Standard Assembly

LangChain provides `create_stuff_documents_chain` which combines documents into context
and passes them to a prompt:

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: Create the document chain
# This chain takes {"context": [list of Documents], "question": str}
# and returns an answer string
document_chain = create_stuff_documents_chain(llm, RAG_PROMPT)

# Step 2: Create the full retrieval chain
# This chain takes {"input": str} and handles retrieval + generation
rag_chain = create_retrieval_chain(retriever, document_chain)

# Step 3: Invoke
result = rag_chain.invoke({"input": "What is the overtime pay rate?"})

print("Answer:")
print(result["answer"])
# "According to the Employee Handbook, overtime is compensated at 1.5x
#  the regular hourly rate for all hours worked beyond 40 in a work week.
#  Sources: [employee-handbook.pdf, page 12]"

print("\nSource documents:")
for doc in result["context"]:
    print(f"  - {doc.metadata['source']}, page {doc.metadata['page']}")
```

`result` contains two keys:

- `"answer"`: the model's generated response (string)
- `"context"`: the list of retrieved `Document` objects (for source display)

---

## Citation Extraction

The generation prompt above requests citations in a specific format.
For programmatic citation extraction, use structured output:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    source_file: str = Field(description="The filename of the source document")
    page_number: Optional[int] = Field(
        default=None,
        description="The page number, if available"
    )

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of sources used to answer the question"
    )
    confidence: str = Field(
        description="How well the context answers the question: 'full', 'partial', 'none'"
    )

# Use structured output for the LLM in the document chain
structured_llm = llm.with_structured_output(AnswerWithCitations)

STRUCTURED_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Answer the question using only the provided context. "
        "List the specific documents and pages you used. "
        "If the context doesn't answer the question, set confidence='none'.\n\n"
        "CONTEXT:\n{context}"
    )),
    ("human", "{question}"),
])

structured_doc_chain = create_stuff_documents_chain(structured_llm, STRUCTURED_RAG_PROMPT)
structured_rag_chain = create_retrieval_chain(retriever, structured_doc_chain)

result = structured_rag_chain.invoke({"input": "What is the overtime pay rate?"})
answer_obj = result["answer"]  # an AnswerWithCitations instance

print(answer_obj.answer)
print(f"Confidence: {answer_obj.confidence}")
for citation in answer_obj.citations:
    print(f"  Source: {citation.source_file}, page {citation.page_number}")
```

---

## Assembling the Full End-to-End Chain

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain(pdf_path: str):
    """
    Full pipeline from PDF to a ready-to-query RAG chain.
    Returns a chain that accepts {"input": question} and returns
    {"answer": str, "context": list[Document]}.
    """
    # --- INDEXING PHASE ---
    # Stage 1: Load
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Stage 2: Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Stage 3: Embed + Stage 4: Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # --- QUERY PHASE ---
    # Stage 5: Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )

    # Stage 6: Generate
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer ONLY using the context below. "
            "Say 'I could not find this information' if context is insufficient.\n\n"
            "CONTEXT:\n{context}"
        )),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# Build once
rag_chain = build_rag_chain("./documents/employee-handbook.pdf")

# Query many times
questions = [
    "What is the overtime pay rate?",
    "How many vacation days do new employees get?",
    "What is the remote work policy?",
]
for q in questions:
    result = rag_chain.invoke({"input": q})
    print(f"\nQ: {q}")
    print(f"A: {result['answer'][:200]}")
```

---

## Common Pitfalls

| Pitfall                                      | What goes wrong                                   | Fix                                                             |
| -------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------- |
| Not grounding the generation prompt          | Model ignores context and uses training knowledge | Explicitly instruct: "Answer ONLY using the provided context"   |
| No "I don't know" instruction                | Model halluccinates when context is absent        | Add "If context doesn't answer, say so" to system prompt        |
| Using `k=1`                                  | Single chunk rarely contains complete answer      | Use `k=4` as default; adjust based on quality                   |
| Not including metadata in context display    | User can't see where the answer came from         | Display `result["context"]` alongside `result["answer"]`        |
| Rebuilding the vector store on every startup | Slow; expensive embedding API calls               | Save FAISS or use Chroma; check if index exists before building |

---

## Mini Summary

- `as_retriever()` converts any vector store into a LangChain Runnable.
- `search_type="similarity"` returns the closest matches; `"mmr"` adds diversity.
- The generation prompt must explicitly instruct the model to use only retrieved context.
- `create_stuff_documents_chain` + `create_retrieval_chain` assemble the full pipeline.
- The result dict contains `"answer"` (string) and `"context"` (retrieved documents).
- Use structured output for citation extraction when you need machine-readable source references.
