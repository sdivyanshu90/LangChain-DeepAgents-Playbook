[← Trace Inspection](03-trace-inspection-and-replay.md) | [Next → Cost and Latency](05-cost-latency-dashboards.md)

---

# 04 — Evaluator Hooks

## Why Programmatic Evaluation?

Manual inspection of agent outputs doesn't scale. Checking 50 outputs by hand takes
hours and introduces human inconsistency. Worse, it's not repeatable: you can't
re-run "human review" after every code change to catch regressions.

Evaluators automate this. An evaluator is a function that receives an agent's input
and output, scores it, and submits the score back to LangSmith. Run the same evaluator
after every change — if the score drops, you know the change caused a regression.

---

## Real-World Analogy

A car manufacturer runs crash tests after every engineering change — not just when
launching a new model. The crash test is the evaluator. The safety rating is the score.
If a door panel redesign reduces the rating from 5 stars to 3, that change is reverted.

Your evaluator is the automated crash test for your agent.

---

## Three Types of Evaluators

```
1. Rule-based (fastest, most reliable for clear criteria)
   ─────────────────────────────────────────────────────
   input: output string
   logic: does it contain required keywords? is it valid JSON? under token limit?
   cost:  zero (no LLM call)
   when:  structural correctness, format compliance

2. LLM-as-judge (flexible, handles nuanced criteria)
   ────────────────────────────────────────────────────────
   input: question + agent output
   logic: another LLM scores on clarity, accuracy, relevance (1-10)
   cost:  one LLM call per evaluation
   when:  quality assessment, tone, completeness

3. Expected output comparison (ground-truth)
   ────────────────────────────────────────────────────────
   input: agent output + expected (gold) answer
   logic: exact match, fuzzy match, or LLM similarity score
   cost:  zero (exact) or one LLM call (semantic)
   when:  factual correctness, regression tests with known answers
```

---

## Submitting Feedback Programmatically

The foundation of all evaluation is `client.create_feedback()`:

```python
from langsmith import Client

client = Client()

def submit_score(
    run_id: str,
    key: str,
    score: float,
    comment: str = "",
) -> None:
    """
    Submit a numeric score for a run to LangSmith.
    Score should be in [0, 1] for consistency.
    """
    client.create_feedback(
        run_id=run_id,
        key=key,              # name of the metric: "correctness", "clarity", etc.
        score=score,           # 0.0 to 1.0
        comment=comment,       # optional human-readable explanation
    )
    print(f"Submitted {key}={score:.2f} for run {run_id[:8]}")

# Example: submit a correctness score after comparing to expected answer
submit_score(
    run_id="your-run-id",
    key="correctness",
    score=1.0,
    comment="Answer matches expected output exactly.",
)
```

---

## Rule-Based Evaluator

```python
def evaluate_json_format(run, example) -> dict:
    """
    Evaluator: checks whether the agent's output is valid JSON.
    Returns LangSmith-compatible feedback dict.
    """
    import json

    output = run.outputs.get("output", "") if run.outputs else ""

    try:
        parsed = json.loads(output)
        score = 1.0
        comment = f"Valid JSON with {len(parsed)} keys" if isinstance(parsed, dict) else "Valid JSON"
    except json.JSONDecodeError as e:
        score = 0.0
        comment = f"Invalid JSON: {e}"

    return {
        "key":     "json_format",
        "score":   score,
        "comment": comment,
    }

def evaluate_length(run, example) -> dict:
    """Evaluator: checks whether output is within the expected length range."""
    output = run.outputs.get("output", "") if run.outputs else ""
    length = len(output)
    MIN_LENGTH = 100
    MAX_LENGTH = 3000

    if length < MIN_LENGTH:
        score, comment = 0.0, f"Too short: {length} chars (min {MIN_LENGTH})"
    elif length > MAX_LENGTH:
        score, comment = 0.5, f"Too long: {length} chars (max {MAX_LENGTH})"
    else:
        score = 1.0
        comment = f"Acceptable length: {length} chars"

    return {"key": "output_length", "score": score, "comment": comment}
```

---

## LLM-as-Judge Evaluator

```python
from langsmith.evaluation import LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def make_llm_judge(criteria: str) -> callable:
    """
    Factory for LLM-as-judge evaluators.
    Each evaluator scores on a specific criterion (1-10, normalised to 0-1).
    """
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    JUDGE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert evaluator. Score the following response on '{criteria}'.\n"
         "Respond ONLY with a JSON object: {{\"score\": <1-10>, \"reason\": \"<one sentence>\"}}"),
        ("human",
         "Question/Prompt: {question}\n\nResponse to evaluate:\n{answer}"),
    ])

    def evaluator(run, example) -> dict:
        import json

        question = example.inputs.get("question", "") if example.inputs else ""
        answer   = run.outputs.get("output", "")      if run.outputs   else ""

        prompt_value = JUDGE_PROMPT.format_messages(
            criteria=criteria,
            question=question,
            answer=answer,
        )
        response = judge_llm.invoke(prompt_value)

        try:
            parsed = json.loads(response.content)
            raw_score = float(parsed["score"])
            normalised = raw_score / 10.0
            comment = parsed.get("reason", "")
        except Exception:
            normalised, comment = 0.5, "Parse failure; defaulting to 0.5"

        return {
            "key":     criteria.lower().replace(" ", "_"),
            "score":   normalised,
            "comment": comment,
        }

    return evaluator

# Create evaluators for different criteria:
clarity_evaluator     = make_llm_judge("clarity")
accuracy_evaluator    = make_llm_judge("factual accuracy")
completeness_evaluator = make_llm_judge("completeness")
```

---

## Correctness Evaluator (Ground Truth)

```python
def evaluate_correctness(run, example) -> dict:
    """
    Compares agent output against an expected (gold) answer.
    Uses semantic similarity for flexible matching.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    import json

    actual   = run.outputs.get("output", "")      if run.outputs   else ""
    expected = example.outputs.get("output", "")  if example.outputs else ""

    if not expected:
        return {"key": "correctness", "score": None, "comment": "No expected output provided"}

    # Exact match (fast, no cost)
    if actual.strip() == expected.strip():
        return {"key": "correctness", "score": 1.0, "comment": "Exact match"}

    # Semantic similarity via LLM
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = judge.invoke([
        HumanMessage(
            f"Rate semantic similarity (0-10) between these two texts.\n"
            f"Respond ONLY with JSON: {{\"score\": <0-10>}}\n\n"
            f"Text A: {expected[:500]}\n\nText B: {actual[:500]}"
        )
    ])

    try:
        score = float(json.loads(response.content)["score"]) / 10.0
    except Exception:
        score = 0.5

    return {
        "key":     "correctness",
        "score":   score,
        "comment": f"Semantic similarity: {score:.2f}",
    }
```

---

## Running Evaluators Against a Dataset

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Assume a dataset "research-agent-test-cases" exists with input/output pairs
DATASET_NAME = "research-agent-test-cases"

# Your agent wrapped as a callable
def run_agent(inputs: dict) -> dict:
    """Wrapper for your LangGraph agent."""
    from langchain_core.messages import HumanMessage
    result = graph.invoke(
        {"messages": [HumanMessage(inputs["question"])]},
        config={"configurable": {"thread_id": f"eval-{inputs.get('id', 'test')}"}},
    )
    return {"output": result["messages"][-1].content}

# Run evaluation
results = evaluate(
    run_agent,
    data=DATASET_NAME,
    evaluators=[
        clarity_evaluator,
        accuracy_evaluator,
        completeness_evaluator,
        evaluate_correctness,
        evaluate_length,
    ],
    experiment_prefix="v3-agent",   # groups results in LangSmith Experiments view
    metadata={"agent_version": "v3.2", "model": "gpt-4o-mini"},
)

print(f"Evaluation complete. View results at:")
print(f"  smith.langchain.com → Projects → {client.read_project(project_name='langchain-deepagents-playbook').id}")
```

---

## Automated Regression Testing

```python
# conftest.py or a dedicated test file
import pytest
from langsmith import Client

client = Client()
REGRESSION_THRESHOLD = 0.75   # minimum acceptable average score

def get_latest_experiment_scores(experiment_prefix: str) -> dict[str, float]:
    """Get average scores for the most recent experiment matching the prefix."""
    # In practice, use client.list_runs(project_name=..., filter=...) to get scores
    # This is a simplified illustration
    return {
        "clarity":     0.85,
        "correctness": 0.82,
        "completeness": 0.78,
    }

@pytest.mark.integration
def test_agent_regression():
    """Fail the build if any evaluator average drops below threshold."""
    scores = get_latest_experiment_scores("v3-agent")
    failing_metrics = {
        key: score
        for key, score in scores.items()
        if score < REGRESSION_THRESHOLD
    }
    assert not failing_metrics, (
        f"Agent regression detected — metrics below {REGRESSION_THRESHOLD}: {failing_metrics}"
    )
```

---

## Common Pitfalls

| Pitfall                                    | Symptom                                          | Fix                                                                                     |
| ------------------------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| LLM judge uses GPT-4o for every evaluation | Evaluation costs more than the agent             | Use `gpt-4o-mini` for evaluation; GPT-4o only for final quality gates                   |
| No ground truth in dataset                 | Can only use LLM-as-judge; no correctness metric | Even rough expected answers are better than none; create them manually for 20+ examples |
| Evaluator raises exception                 | Evaluation run crashes; no score recorded        | Wrap evaluator body in try/except; return `{"score": None}` on failure                  |
| Score not normalised to [0, 1]             | LangSmith charts are inconsistent                | Always normalise: divide 1-10 scales by 10                                              |
| Only evaluating on failure cases           | Dataset is unbalanced; score is inflated         | Include both passing and failing examples in your dataset                               |

---

## Mini Summary

- Evaluators are functions that take `(run, example) → {"key": str, "score": float}` and submit scores to LangSmith
- Three types: rule-based (fast, free), LLM-as-judge (flexible, one LLM call), and correctness (ground truth)
- `client.create_feedback()` submits scores programmatically; `evaluate()` runs them against a full dataset
- Automated regression tests compare evaluation scores against a threshold after every code change
- Use `gpt-4o-mini` for evaluation to keep evaluation costs low; reserve better models for the agent itself

---

[← Trace Inspection](03-trace-inspection-and-replay.md) | [Next → Cost and Latency](05-cost-latency-dashboards.md)
