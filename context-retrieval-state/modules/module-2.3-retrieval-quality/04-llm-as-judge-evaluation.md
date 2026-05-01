# 04 — LLM-as-Judge Evaluation

> **Previous:** [03 → Chunking Experiments](03-chunking-experiments.md) | **Next:** [05 → Citations and Grounding](05-citations-and-grounding.md)

---

## Real-World Analogy

A restaurant doesn't only ask the chef if the food is good.
It hires independent reviewers who don't know who cooked it.
Objective evaluation requires a judge who is not the same system that produced the output.

In RAG evaluation, the "judge" is another LLM instance — separate from the one answering.
It scores the answers against known-good references.

---

## Why You Need Systematic Evaluation

Informal testing — "I tried 10 questions and they seemed fine" — is not evaluation.
It creates false confidence.

```
Without evaluation:
  Developer tests: "What is the overtime rate?" → correct → ships
  Production: 1000 questions/day
    → 18% are paraphrasings the developer didn't test
    → 12% of those get wrong chunks
    → ~50 wrong answers per day
    → discovered 3 weeks later

With evaluation:
  Developer builds eval dataset: 50 diverse questions
  Runs eval before and after each change
  Baseline: 0.78 context relevance
  After MultiQueryRetriever: 0.91 context relevance
  Ships with confidence and clear improvement metric
```

---

## The Three Core RAG Metrics

Before writing code, understand what you're measuring:

```
1. Context Relevance
───────────────────
  "Did retrieval return the right chunks?"
  Measures: are the retrieved chunks relevant to the question?
  Perfect score: every retrieved chunk is directly relevant.
  Low score: wrong sections were retrieved.

2. Faithfulness
───────────────
  "Is the answer supported by the retrieved context?"
  Measures: does every claim in the answer appear in the retrieved context?
  Perfect score: all claims traceable to context (no hallucination).
  Low score: model added facts not in the retrieved chunks.

3. Answer Relevance
───────────────────
  "Does the answer actually address the question?"
  Measures: is the answer on-topic and complete?
  Perfect score: answers the question fully.
  Low score: answer is technically correct but misses the point.
```

---

## LLM-as-Judge: Simple 1-10 Scoring

Before using a framework, understand the pattern with a bare implementation:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# ↑ Use the same or a stronger model as a judge.
# temperature=0: deterministic scoring is critical for reproducible evaluation.

class RelevanceScore(BaseModel):
    """Score assigned by the judge LLM for a single retrieval result."""
    score: int = Field(
        ge=1, le=10,
        description="Relevance of the retrieved context to the question, 1-10"
    )
    reasoning: str = Field(
        description="Brief explanation of why this score was assigned"
    )

judge_chain = (
    ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert evaluator of RAG (Retrieval Augmented Generation) systems.\n"
            "Score how relevant the provided CONTEXT is for answering the QUESTION.\n\n"
            "Scoring guide:\n"
            "10: Context directly and completely answers the question.\n"
            "7-9: Context contains the answer but with some irrelevant information.\n"
            "4-6: Context is topically related but doesn't fully answer the question.\n"
            "1-3: Context is off-topic or contains misleading information.\n\n"
            "Be strict. A score of 8+ means the context is genuinely useful."
        )),
        ("human", (
            "QUESTION: {question}\n\n"
            "CONTEXT:\n{context}\n\n"
            "Assign a relevance score from 1 to 10."
        )),
    ])
    | judge_llm.with_structured_output(RelevanceScore)
)

def score_retrieval(question: str, retrieved_docs: list) -> float:
    """
    Score retrieval relevance using LLM-as-judge.
    Returns the average score across all retrieved chunks, normalised to 0-1.
    """
    scores = []
    for doc in retrieved_docs:
        result = judge_chain.invoke({
            "question": question,
            "context": doc.page_content,
        })
        scores.append(result.score)
        # Optional: log result.reasoning for debugging
    return sum(scores) / len(scores) / 10.0   # normalise to 0.0-1.0
```

---

## Building an Evaluation Dataset

An eval dataset is the foundation of reliable measurement.
Build it before optimising — it's the baseline you compare against.

```python
import json
from pathlib import Path
from typing import TypedDict

class EvalSample(TypedDict):
    """One evaluation sample: question + expected answer for grading."""
    question:         str
    expected_answer:  str
    source_document:  str      # which document should be retrieved
    source_page:      int      # which page the answer is on

# Build the eval dataset manually from your documents
# Aim for: 30-50 samples, diverse question types
EVAL_DATASET: list[EvalSample] = [
    {
        "question": "What is the overtime pay rate for non-exempt employees?",
        "expected_answer": "1.5x regular hourly rate for hours beyond 40 per week",
        "source_document": "employee-handbook.pdf",
        "source_page": 12,
    },
    {
        "question": "How many days of sick leave does a full-time employee receive annually?",
        "expected_answer": "10 days per year, non-cumulative",
        "source_document": "employee-handbook.pdf",
        "source_page": 15,
    },
    {
        "question": "What is the notice period for resignation?",
        "expected_answer": "2 weeks for individual contributors, 4 weeks for managers",
        "source_document": "employee-handbook.pdf",
        "source_page": 31,
    },
    # Add 30-50 samples covering different sections and question types
]

# Save to disk for reproducibility
dataset_path = Path("./eval_dataset.json")
dataset_path.write_text(json.dumps(EVAL_DATASET, indent=2))
print(f"Saved {len(EVAL_DATASET)} eval samples to {dataset_path}")
```

---

## Running a Full Evaluation

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def evaluate_rag_system(
    retriever,
    eval_dataset: list[EvalSample],
    run_name: str = "baseline",
) -> dict:
    """
    Run the evaluation dataset against a retriever.
    Returns summary statistics and per-sample scores.
    """
    results = []

    for sample in eval_dataset:
        question = sample["question"]
        expected = sample["expected_answer"]

        # Get retrieved chunks
        retrieved_docs = retriever.invoke(question)

        # Score 1: Context relevance (LLM judge)
        context_relevance = score_retrieval(question, retrieved_docs)

        # Score 2: Answer faithfulness (was expected answer found in context?)
        context_text = "\n".join(d.page_content for d in retrieved_docs)
        faithfulness_result = judge_chain.invoke({
            "question": f"Does this context support the answer: '{expected}'?",
            "context": context_text,
        })
        faithfulness = faithfulness_result.score / 10.0

        # Score 3: Did we retrieve from the right source?
        expected_source = sample["source_document"]
        sources_retrieved = [d.metadata.get("source", "") for d in retrieved_docs]
        source_hit = any(expected_source in s for s in sources_retrieved)

        results.append({
            "question":          question,
            "context_relevance": context_relevance,
            "faithfulness":      faithfulness,
            "source_hit":        source_hit,
        })

    # Summary statistics
    summary = {
        "run_name":              run_name,
        "n_samples":             len(results),
        "avg_context_relevance": sum(r["context_relevance"] for r in results) / len(results),
        "avg_faithfulness":      sum(r["faithfulness"]      for r in results) / len(results),
        "source_hit_rate":       sum(r["source_hit"]        for r in results) / len(results),
    }

    print(f"\n{'='*50}")
    print(f"Evaluation: {run_name}")
    print(f"  Context Relevance: {summary['avg_context_relevance']:.3f}")
    print(f"  Faithfulness:      {summary['avg_faithfulness']:.3f}")
    print(f"  Source Hit Rate:   {summary['source_hit_rate']:.3f}")

    return {"summary": summary, "per_sample": results}
```

---

## RAGAS Framework — Introduction

RAGAS is a dedicated framework for RAG evaluation.
It implements the three core metrics with research-grade methods:

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,    # are retrieved chunks relevant? (context relevance)
    faithfulness,         # are claims in the answer supported by context?
    answer_relevancy,     # does the answer address the question?
    context_recall,       # were all relevant chunks retrieved? (needs reference answer)
)
from datasets import Dataset

# RAGAS expects a dataset with specific column names
ragas_data = {
    "question":  [s["question"]        for s in EVAL_DATASET],
    "answer":    [],    # filled after running your RAG chain
    "contexts":  [],    # filled with retrieved chunk texts
    "ground_truth": [s["expected_answer"] for s in EVAL_DATASET],
}

# Run your RAG chain on all questions and collect answers + contexts
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

for sample in EVAL_DATASET:
    rag_result = rag_chain.invoke({"input": sample["question"]})
    ragas_data["answer"].append(rag_result["answer"])
    ragas_data["contexts"].append(
        [doc.page_content for doc in rag_result["context"]]
    )

# Run RAGAS evaluation
dataset = Dataset.from_dict(ragas_data)
score = evaluate(
    dataset=dataset,
    metrics=[context_precision, faithfulness, answer_relevancy],
)

print(score)
# {'context_precision': 0.87, 'faithfulness': 0.91, 'answer_relevancy': 0.83}
```

---

## Using Evaluation to Drive Improvement

```
Baseline (simple similarity, chunk_size=512):
  context_precision: 0.78
  faithfulness:      0.85

After MultiQueryRetriever:
  context_precision: 0.89  (+14%)  ← vocabulary mismatch resolved
  faithfulness:      0.85  (same)  ← not affected

After ParentDocumentRetriever:
  context_precision: 0.91  (+3%)
  faithfulness:      0.92  (+8%)   ← more complete context → fewer hallucinations

Decision: ship with MultiQuery + ParentDocument.
          The additional cost is justified by the improvement.
```

---

## Common Pitfalls

| Pitfall                                             | What goes wrong                                             | Fix                                                          |
| --------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Using the same LLM to generate and judge answers    | Judge may be biased toward its own style                    | Use a different model or prompt as judge, or use RAGAS       |
| Eval dataset too small (< 20 samples)               | Results are noisy; can't detect real improvements           | Build 30-50 samples minimum; 100 is better                   |
| Only measuring final answer quality                 | You don't know if retrieval or generation is the bottleneck | Measure context relevance and faithfulness separately        |
| Re-using eval questions in prompt few-shot examples | Judge "recognises" the questions; inflated scores           | Strict separation: eval set must not appear in training data |
| Not versioning eval results                         | You can't compare before/after improvement                  | Save eval results with timestamps and config labels          |

---

## Mini Summary

- LLM-as-judge uses a separate model instance to score retrieval and answer quality.
- Three core metrics: context relevance (right chunks?), faithfulness (grounded?), answer relevance (on-topic?).
- Build your eval dataset before optimising — it is the baseline you compare against.
- Run evaluation before and after each change to measure real improvement.
- RAGAS provides research-grade metrics; use it when you need more rigorous evaluation.
- Eval set must be kept separate from any few-shot examples or prompt engineering.
