[← Evaluator Hooks](04-evaluator-hooks.md) | [← Back to Module Overview](README.md)

---

# 05 — Cost and Latency Dashboards

## Why Cost Tracking Matters

An agent that works correctly but costs $2 per run at 10,000 daily users costs
$20,000 per day — $7.3 million per year. Cost tracking is not optional for
production agents; it's a financial control.

Latency tracking is equally important: users abandon AI features that take more than
3-5 seconds. Identifying the slow node is the first step to fixing it.

LangSmith captures both automatically. This file shows you how to read dashboards,
set alerts, and track costs locally when LangSmith is unavailable.

---

## Real-World Analogy

A restaurant kitchen has two dashboards: food cost per dish and table service time.
The head chef uses both to optimise the menu and kitchen workflow. A dish that costs
$18 in ingredients but sells for $22 is replaced. A dish that takes 40 minutes to
plate during rush hour is simplified. The dashboards make invisible problems visible.

Your agent's cost and latency dashboards serve the same purpose.

---

## Token Cost Calculation

Pricing as of June 2025 (verify at platform.openai.com/docs/pricing):

```python
# Cost reference — update when pricing changes
MODEL_PRICING = {
    "gpt-4o": {
        "prompt":     0.005   / 1000,   # $0.005 per 1K prompt tokens
        "completion": 0.015   / 1000,   # $0.015 per 1K completion tokens
    },
    "gpt-4o-mini": {
        "prompt":     0.00015 / 1000,   # $0.00015 per 1K prompt tokens
        "completion": 0.0006  / 1000,   # $0.0006 per 1K completion tokens
    },
    "text-embedding-3-small": {
        "prompt":     0.00002 / 1000,   # embeddings; no completion
        "completion": 0.0,
    },
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost for a single LLM call."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    return (
        prompt_tokens     * pricing["prompt"]
        + completion_tokens * pricing["completion"]
    )

# Example:
cost = calculate_cost("gpt-4o-mini", prompt_tokens=500, completion_tokens=150)
print(f"Cost: ${cost:.6f}")   # → Cost: $0.000165
```

---

## Reading Token Usage in LangChain Responses

Every ChatOpenAI response includes `response_metadata` with token counts:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([HumanMessage("Summarise the CAP theorem in 3 sentences.")])

# Token usage is in response_metadata
usage = response.response_metadata.get("token_usage", {})
prompt_tokens     = usage.get("prompt_tokens", 0)
completion_tokens = usage.get("completion_tokens", 0)
total_tokens      = usage.get("total_tokens", 0)

cost = calculate_cost("gpt-4o-mini", prompt_tokens, completion_tokens)

print(f"Prompt tokens:     {prompt_tokens}")
print(f"Completion tokens: {completion_tokens}")
print(f"Total tokens:      {total_tokens}")
print(f"Cost:              ${cost:.6f}")
```

---

## Local Cost Tracking Without LangSmith

For environments where LangSmith is disabled (unit tests, CI, offline development),
track costs in a local accumulator:

```python
import time
from dataclasses import dataclass, field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

@dataclass
class CostTracker:
    """Accumulates cost and latency data for a single agent run."""
    runs:           list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens:   int   = 0
    total_latency_s: float = 0.0

    def log(self, node: str, model: str, prompt_tokens: int,
            completion_tokens: int, latency_s: float) -> None:
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
        self.runs.append({
            "node":              node,
            "model":             model,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd":          cost,
            "latency_s":         latency_s,
        })
        self.total_cost_usd  += cost
        self.total_tokens    += prompt_tokens + completion_tokens
        self.total_latency_s += latency_s

    def report(self) -> None:
        print(f"\n{'='*55}")
        print(f"Run Cost Report")
        print(f"{'='*55}")
        print(f"{'Node':25} {'Tokens':>8} {'Cost':>10} {'Latency':>10}")
        print(f"{'-'*55}")
        for r in self.runs:
            print(f"{r['node']:25} {r['prompt_tokens'] + r['completion_tokens']:>8} "
                  f"${r['cost_usd']:>9.6f} {r['latency_s']:>9.2f}s")
        print(f"{'-'*55}")
        print(f"{'TOTAL':25} {self.total_tokens:>8} "
              f"${self.total_cost_usd:>9.6f} {self.total_latency_s:>9.2f}s")
        print(f"{'='*55}\n")

# Integration in a node:
tracker = CostTracker()

def instrumented_node(state, config=None) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini")
    start = time.perf_counter()
    response = llm.invoke(state["messages"])
    latency = time.perf_counter() - start

    usage = response.response_metadata.get("token_usage", {})
    tracker.log(
        node="instrumented_node",
        model="gpt-4o-mini",
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_s=latency,
    )
    return {"messages": [response]}
```

---

## LangSmith Cost Dashboard

In LangSmith, navigate to:
`Projects → [your project] → Monitoring`

Key views:

```
Monitoring Dashboard
├── Total Cost Over Time      — daily/weekly cost trend; spot sudden spikes
├── Cost by Run Name          — which agent type costs most
├── Cost by Tag               — production vs staging comparison
├── Token Usage by Model      — prompt vs completion ratio (high prompt → cache opportunity)
├── Latency Percentiles       — p50, p95, p99 — p99 reveals worst-case user experience
└── Latency by Node           — which node is the bottleneck
```

**Reading the latency percentile chart:**

- p50 = median latency (typical user experience)
- p95 = 95th percentile (most users experience this or better)
- p99 = 99th percentile (worst 1% of requests)

If p50 is 1.5s but p99 is 12s, some users have terrible experiences. This points to
a non-deterministic slow path (e.g., a tool that occasionally times out).

---

## Setting Up Cost Alerts

LangSmith does not natively send email alerts for cost thresholds (as of mid-2025),
but you can implement this with a scheduled check:

```python
from langsmith import Client
from datetime import datetime, timedelta, timezone

client = Client()

DAILY_COST_ALERT_USD = 5.00   # alert if single-day cost exceeds this

def check_daily_cost(project_name: str) -> None:
    """
    Check total cost for the past 24 hours.
    Alert (print/email/Slack) if above threshold.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    runs = list(client.list_runs(
        project_name=project_name,
        start_time=since,
    ))

    # LangSmith stores cost in run.total_cost (USD)
    total_cost = sum(
        r.total_cost or 0.0
        for r in runs
        if r.total_cost is not None
    )

    print(f"Last 24h cost for '{project_name}': ${total_cost:.4f}")

    if total_cost > DAILY_COST_ALERT_USD:
        alert_message = (
            f"COST ALERT: ${total_cost:.4f} spent in the last 24 hours "
            f"(threshold: ${DAILY_COST_ALERT_USD:.2f}) "
            f"across {len(runs)} runs."
        )
        print(f"[ALERT] {alert_message}")
        # In production: send to Slack, PagerDuty, email, etc.

# Run as a cron job (e.g., every hour):
# check_daily_cost("langchain-deepagents-playbook")
```

---

## Identifying the Bottleneck Node

```python
from langsmith import Client
from collections import defaultdict

client = Client()

def identify_bottleneck(project_name: str, run_name_prefix: str) -> None:
    """
    Analyse recent runs to find which node has the highest average latency.
    """
    runs = list(client.list_runs(
        project_name=project_name,
        filter=f'eq(name, "{run_name_prefix}")',
        limit=50,
    ))

    node_latencies: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        child_runs = list(client.list_runs(parent_run_id=run.id))
        for child in child_runs:
            if child.start_time and child.end_time:
                latency = (child.end_time - child.start_time).total_seconds()
                node_latencies[child.name].append(latency)

    print(f"\nLatency analysis for '{run_name_prefix}' (last {len(runs)} runs):")
    print(f"{'Node':35} {'Count':>6} {'Avg (s)':>10} {'Max (s)':>10}")
    print("-" * 65)

    sorted_nodes = sorted(
        node_latencies.items(),
        key=lambda x: sum(x[1]) / len(x[1]),
        reverse=True,
    )
    for name, latencies in sorted_nodes:
        avg = sum(latencies) / len(latencies)
        max_l = max(latencies)
        print(f"{name:35} {len(latencies):>6} {avg:>10.2f} {max_l:>10.2f}")
```

---

## Cost Per Session Analysis

```python
def analyse_cost_per_session(
    project_name: str,
    hours: int = 24,
) -> dict[str, float]:
    """
    Calculate total cost grouped by session_id metadata.
    Identifies high-cost sessions for investigation.
    """
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    runs = list(client.list_runs(
        project_name=project_name,
        start_time=since,
    ))

    session_costs: dict[str, float] = defaultdict(float)
    for run in runs:
        session_id = (run.extra or {}).get("metadata", {}).get("session_id", "unknown")
        session_costs[session_id] += run.total_cost or 0.0

    # Sort by cost descending
    sorted_sessions = sorted(session_costs.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop sessions by cost (last {hours}h):")
    for session_id, cost in sorted_sessions[:10]:
        print(f"  {session_id:40} ${cost:.6f}")

    return dict(sorted_sessions)
```

---

## Common Pitfalls

| Pitfall                                            | Symptom                            | Fix                                                                                                          |
| -------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Tracking cost in loop with `+=` but not persisting | Cost resets on process restart     | Write cost data to a file/DB, or rely on LangSmith                                                           |
| Missing `response_metadata` in some models         | `token_usage` is `{}`              | Use OpenAI models which always return token usage; check `model_kwargs={"stream_usage": True}` for streaming |
| p50 latency looks good; ignoring p99               | Tail latency problems missed       | Always check p99; it reveals timeout/retry issues                                                            |
| No session_id in metadata                          | Can't group runs by user session   | Add `session_id` to every run's metadata (Module 3.6/02)                                                     |
| Cost alert threshold too low for dev               | False positives during development | Use separate projects for `production` and `development`; set thresholds per project                         |

---

## Mini Summary

- Every LangChain response includes `response_metadata["token_usage"]` — use it to track costs locally
- LangSmith Monitoring dashboard shows cost over time, cost by tag, and latency percentiles (p50/p95/p99)
- The bottleneck node is always in the p99 latency tail — don't optimise based on average latency alone
- Use `client.list_runs()` with metadata filters to calculate cost-per-session for billing attribution
- Set up a scheduled cost check against LangSmith's `run.total_cost` to alert on unexpected spending spikes

---

[← Evaluator Hooks](04-evaluator-hooks.md) | [← Back to Module Overview](README.md)
