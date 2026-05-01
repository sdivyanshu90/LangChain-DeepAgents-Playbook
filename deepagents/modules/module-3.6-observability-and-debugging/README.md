# Module 3.6 — Observability and Debugging

> **Why this module exists:** A LangGraph agent without observability is a black box.
> When it fails, you can only guess why. When it's slow, you can't tell which node is
> the bottleneck. When it's expensive, you can't see where the tokens are going.
> LangSmith turns the black box into a glass box: full trace trees, latency breakdowns,
> token costs, and programmatic evaluation — all with zero code changes to your agent.

---

## Topics

| #   | File                                                             | What you will learn                                                                 |
| --- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 01  | [LangSmith Setup](01-langsmith-setup.md)                         | Environment variables, free tier limits, enabling per-run, `run_name` and `tags`    |
| 02  | [Run Metadata and Tags](02-run-metadata-and-tags.md)             | Adding metadata to runs and nodes; `RunnableConfig`; filtering in LangSmith UI      |
| 03  | [Trace Inspection and Replay](03-trace-inspection-and-replay.md) | Reading the trace tree; identifying slow nodes; replaying failed runs               |
| 04  | [Evaluator Hooks](04-evaluator-hooks.md)                         | Custom evaluators; LLM-as-judge; dataset + evaluator for regression testing         |
| 05  | [Cost and Latency Dashboards](05-cost-latency-dashboards.md)     | Token cost per run; latency per node; cost alerts; local tracking without LangSmith |

---

## What You Can See With LangSmith vs Without

```
WITHOUT LANGSMITH                    WITH LANGSMITH
─────────────────────────────────    ─────────────────────────────────────────────
                                     ┌─ Run: "research-task-001" ──────────────┐
User input                           │  Total: 4.2s | $0.0034 | ✓ success      │
     │                               │                                          │
     ▼                               │  ├─ planner_node            0.8s  $0.0012│
 [AGENT]                             │  │   input: "analyse AI..."              │
     │                               │  │   output: {plan: [7 steps]}           │
     ▼                               │  │                                       │
Final answer                         │  ├─ executor_node (×5)      2.8s  $0.0018│
(no intermediate info)               │  │   ├─ step 1: search_web  0.3s         │
                                     │  │   ├─ step 2: search_web  0.6s  ← slow │
If it fails:                         │  │   ├─ step 3: calculate   0.1s         │
  → No idea which node failed        │  │   ├─ step 4: search_web  0.9s  ← slow │
  → No idea what input caused it     │  │   └─ step 5: search_web  0.9s  ← slow │
  → No idea how much it cost         │  │                                       │
  → Cannot replay the failure        │  └─ synthesiser_node        0.6s  $0.0004│
                                     └──────────────────────────────────────────┘

                                     When it fails:
                                       → Exact node that failed
                                       → Full input at that node
                                       → Error message and stack trace
                                       → Replay with corrected input
```

---

## Key Environment Variables

```bash
# Required
LANGSMITH_API_KEY=lsv2_...          # from smith.langchain.com
LANGSMITH_TRACING=true              # enable tracing

# Optional but recommended
LANGSMITH_PROJECT=my-project-name   # organise runs by project
LANGSMITH_ENDPOINT=https://api.smith.langchain.com  # default; change for self-hosted

# OpenAI (also required for the agents)
OPENAI_API_KEY=sk-...
```
