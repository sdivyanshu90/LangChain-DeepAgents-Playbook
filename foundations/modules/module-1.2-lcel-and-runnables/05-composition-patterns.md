# 05 — Composition Patterns

> **Previous:** [04 → Data Shape Discipline](04-data-shape-discipline.md) | **Next:** [Module 1.3 → Structured Output](../module-1.3-output-formatting-and-structured-responses/README.md)

---

## Overview

This topic is a pattern cookbook.
Each pattern has a name, a diagram, annotated code, and the specific problem it solves.
Return to this file as a reference whenever you design a new chain.

---

## Pattern 1 — Linear Pipeline

**Problem:** Transform input through a deterministic sequence of steps.

```
input → A → B → C → output
```

```python
chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)
result = chain.invoke({"question": "What is entropy?"})
```

**Use when:** The transformation is purely sequential with no branching or parallel steps.

---

## Pattern 2 — Fan-Out (Parallel Analysis)

**Problem:** Generate multiple independent views of the same input simultaneously.

```
            ┌──→ summary_chain  → "One sentence..."
input ──────┤
            ├──→ risks_chain    → ["risk1", ...]
            │
            └──→ questions_chain → ["Q1?", "Q2?"]
```

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

analysis = RunnableParallel(
    original  = RunnablePassthrough(),
    summary   = summary_prompt   | llm | StrOutputParser(),
    risks     = risks_prompt     | llm | JsonOutputParser(),
    questions = questions_prompt | llm | JsonOutputParser(),
)

result = analysis.invoke({"text": article_text})
# result["summary"]   → str
# result["risks"]     → list
# result["questions"] → list
# result["original"]  → {"text": article_text}
```

**Use when:** Multiple outputs from one input; each is independent; speed matters.

---

## Pattern 3 — Sequential Enrichment

**Problem:** Build a rich context dict step by step, where each step can use everything added so far.

```
{"text": "..."}
    ↓  .assign(language=...)
{"text": "...", "language": "en"}
    ↓  .assign(summary=...)
{"text": "...", "language": "en", "summary": "..."}
    ↓  .assign(tags=...)
{"text": "...", "language": "en", "summary": "...", "tags": [...]}
```

```python
from langchain_core.runnables import RunnablePassthrough

enrich = (
    RunnablePassthrough.assign(
        language=RunnableLambda(lambda x: detect_language(x["text"]))
    )
    .assign(
        summary=RunnableLambda(lambda x: {
            "text": x["text"],
            "language": x["language"]
        }) | summary_chain
    )
    .assign(
        tags=tags_chain  # receives full dict with text + language + summary
    )
)

result = enrich.invoke({"text": "Bonjour le monde..."})
```

**Use when:** Later steps need the output of earlier steps; the result must accumulate all fields.

---

## Pattern 4 — Context Injection (RAG Core)

**Problem:** Combine a user question with retrieved context before asking the model.

```
question ──────────────────────────────────┐
                                            ├──→ answer_prompt → llm → parser
question → retriever → format_docs ────────┘
```

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

rag_chain = (
    {
        "context":  itemgetter("question") | retriever | RunnableLambda(format_docs),
        "question": itemgetter("question"),
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke({"question": "What is the refund policy?"})
```

**Use when:** Building any RAG pipeline. This is the standard pattern for Module 2.2+.

---

## Pattern 5 — Conditional Branching

**Problem:** Route to different sub-chains based on a classification of the input.

```
input → classifier
            ├── "technical" → technical_chain
            ├── "billing"   → billing_chain
            └── (default)   → general_chain
```

```python
from langchain_core.runnables import RunnableBranch

# Step 1: Classify first (add to dict)
classify = RunnablePassthrough.assign(
    category=category_classifier_chain  # returns "technical" | "billing" | "general"
)

# Step 2: Branch based on classification
route = RunnableBranch(
    (lambda x: x["category"] == "technical", technical_chain),
    (lambda x: x["category"] == "billing",   billing_chain),
    general_chain,  # default
)

# Step 3: Compose
full_chain = classify | route

result = full_chain.invoke({"question": "My invoice is wrong."})
# → classified as "billing" → billing_chain handles it
```

**Use when:** The routing condition is simple and stateless. For multi-turn routing or loops, use LangGraph.

---

## Pattern 6 — Map-Reduce

**Problem:** Process a list of items, then aggregate the results.

```
["doc1", "doc2", "doc3"]
        │
        ├──→ summarize(doc1) → "s1"
        ├──→ summarize(doc2) → "s2"    (parallel)
        └──→ summarize(doc3) → "s3"
                    │
                    ▼
        ["s1", "s2", "s3"]
                    │
                    ▼
           synthesize → "final answer"
```

```python
from langchain_core.runnables import RunnableLambda

def map_summaries(docs: list[str]) -> list[str]:
    """Summarise each document in parallel."""
    inputs = [{"text": d} for d in docs]
    return summarize_chain.batch(inputs, config={"max_concurrency": 5})

def reduce_summaries(summaries: list[str]) -> str:
    """Synthesise all summaries into one answer."""
    combined = "\n\n".join(f"- {s}" for s in summaries)
    return synthesize_chain.invoke({"summaries": combined})

map_reduce_chain = (
    RunnableLambda(map_summaries)
    | RunnableLambda(reduce_summaries)
)

result = map_reduce_chain.invoke(documents)
```

**Use when:** Summarising a collection of documents, grading a batch of answers, aggregating search results.

---

## Pattern 7 — Self-Critique Loop (Preview of Reflexion)

**Problem:** Generate an output, evaluate it, and regenerate if quality is insufficient.

```
input → generator → output_v1
                        │
                    evaluator → score
                        │
                    score >= threshold?
                        ├── Yes → return output_v1
                        └── No  → generator (with critique) → output_v2
```

```python
from langchain_core.runnables import RunnableLambda

MAX_ATTEMPTS = 3

def generate_with_critique(state: dict) -> dict:
    attempt = state.get("attempt", 0)
    critique = state.get("critique", "")

    prompt_input = {
        "task": state["task"],
        "critique": f"Prior attempt failed: {critique}" if critique else "",
    }
    output = generator_chain.invoke(prompt_input)
    score = evaluator_chain.invoke({"output": output})

    return {
        **state,
        "output": output,
        "score": score,
        "attempt": attempt + 1,
    }

def should_continue(state: dict) -> dict:
    if state["score"] >= 8.0 or state["attempt"] >= MAX_ATTEMPTS:
        return state  # done
    # Add critique for the next iteration
    state["critique"] = f"Score {state['score']}/10. Improve: ..."
    return generate_with_critique(state)

loop_chain = RunnableLambda(generate_with_critique) | RunnableLambda(should_continue)
```

**Note:** For production self-critique loops, use LangGraph (Module 3.4) which handles
state persistence and explicit loop guards more cleanly.

---

## Composing Patterns Together

Real chains combine multiple patterns:

```
FAQ Generator chain (Project B-3):

input ─→ detect_language (Pattern 1 / Lambda)
       ─→ extract_qa_pairs (Pattern 1 / structured output)
       ─→ categorise_entries (Pattern 2 / parallel)
       ─→ confidence_filter (Pattern 5 / branch: flag low-confidence)
       ─→ format_output (Pattern 1 / RunnableLambda)
```

---

## Pattern Selection Guide

```
One input → one output, sequential?          → Pattern 1 (Linear)
One input → multiple independent outputs?    → Pattern 2 (Fan-Out)
Each step needs prior step's output?         → Pattern 3 (Sequential Enrichment)
Inject retrieved context into a prompt?      → Pattern 4 (Context Injection)
Route to different chains by condition?      → Pattern 5 (Conditional Branch)
Apply same chain to a list, then aggregate?  → Pattern 6 (Map-Reduce)
Generate → evaluate → improve?               → Pattern 7 (Self-Critique)
```

---

## Mini Summary

- Seven patterns cover the vast majority of real LangChain chain designs.
- Patterns 1–5 are Level 1; Patterns 6–7 preview Level 2–3 needs.
- Real chains are composites: identify the pattern for each sub-problem, then assemble.
- When a pattern requires loops, persistent state, or complex routing, graduate to LangGraph.

---

## Next: [Module 1.3 → Structured Output and Schema Design](../module-1.3-output-formatting-and-structured-responses/README.md)
