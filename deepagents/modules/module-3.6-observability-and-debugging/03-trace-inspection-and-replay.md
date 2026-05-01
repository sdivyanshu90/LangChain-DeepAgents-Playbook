[← Run Metadata](02-run-metadata-and-tags.md) | [Next → Evaluator Hooks](04-evaluator-hooks.md)

---

# 03 — Trace Inspection and Replay

## Why Inspection and Replay?

When an agent fails in production, you need to answer three questions:

1. Which node failed?
2. What input did that node receive?
3. Can I reproduce the failure with a fixed prompt?

Without a trace, answering question 1 takes a long time. Questions 2 and 3 may be
impossible — the inputs were transient and are now gone.

LangSmith trace inspection answers question 1 in seconds (click the red span).
Questions 2 and 3 are answered by the "Open in Playground" and "Add to Dataset"
features — you can replay any historical run with modified inputs.

---

## Real-World Analogy

Airline maintenance engineers use flight data recorder (FDR) data not just to
investigate crashes, but to proactively identify sensor anomalies before they
cause failures. They replay specific flight segments in simulation to test whether
a proposed fix would have changed the outcome.

LangSmith trace replay is the "replay in simulation" step.

---

## Reading the LangSmith Trace Tree

Every LangGraph run produces a hierarchical trace:

```
graph_run ("research-task-001")                 4.2s   $0.0034
├── __start__ (graph entry)                     0.0s
├── planner_node                                0.8s   $0.0012
│   ├── ChatOpenAI (gpt-4o)                     0.8s   $0.0012
│   │   input:  [SystemMessage("You are..."), HumanMessage("Goal: analyse...")]
│   │   output: AIMessage('{"goal": ..., "steps": [...]}')
│   │   tokens: prompt=312, completion=218
│   │   latency: 0.8s
│   └── (end)
├── executor_node (iteration=1)                 0.3s   $0.0004
│   ├── ChatOpenAI (gpt-4o-mini)                0.3s   $0.0004
│   │   ...
│   └── tool: search_web                        0.1s
│       input:  {"query": "AI coding assistants 2025"}
│       output: "Result: GitHub Copilot..."
├── executor_node (iteration=2)                 0.9s   $0.0006  ← slow
│   ...
└── synthesiser_node                            0.6s   $0.0004
    ...
```

**Key columns in LangSmith:**

- **Latency**: total time for each span
- **Tokens**: prompt + completion (hover for breakdown)
- **Cost**: USD estimate (based on model pricing)
- **Status**: green = success, red = error, yellow = warning

---

## Identifying Slow Nodes

```python
# Programmatic latency analysis using the LangSmith SDK
from langsmith import Client

client = Client()

def analyse_run_latency(run_id: str) -> None:
    """Print latency breakdown for all nodes in a run."""
    run = client.read_run(run_id)
    child_runs = list(client.list_runs(parent_run_id=run_id))

    print(f"Total latency: {run.end_time - run.start_time}")
    print("\nNode breakdown:")

    for child in sorted(child_runs, key=lambda r: r.name):
        if child.start_time and child.end_time:
            latency = (child.end_time - child.start_time).total_seconds()
            print(f"  {child.name:35} {latency:6.2f}s")

# Usage (replace with your actual run ID from LangSmith UI):
# analyse_run_latency("run-id-from-langsmith")
```

---

## Comparing Two Runs

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

def compare_runs(run_id_1: str, run_id_2: str) -> None:
    """
    Compare latency and cost between two runs.
    Useful for before/after prompt optimisation comparisons.
    """
    run_1 = client.read_run(run_id_1)
    run_2 = client.read_run(run_id_2)

    def run_stats(run) -> dict:
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time else 0
        tokens  = run.total_tokens or 0
        return {"latency_s": latency, "total_tokens": tokens}

    s1 = run_stats(run_1)
    s2 = run_stats(run_2)

    print(f"{'Metric':20} {'Run 1':>10} {'Run 2':>10} {'Delta':>10}")
    print("-" * 55)
    print(f"{'Latency (s)':20} {s1['latency_s']:>10.2f} {s2['latency_s']:>10.2f} "
          f"{s2['latency_s'] - s1['latency_s']:>+10.2f}")
    print(f"{'Total tokens':20} {s1['total_tokens']:>10} {s2['total_tokens']:>10} "
          f"{s2['total_tokens'] - s1['total_tokens']:>+10}")
```

---

## Replaying Failed Runs

### Option 1: Via LangSmith UI (simplest)

1. Find the failed run in the trace list
2. Click the failed span
3. Click **"Open in Playground"**
4. Modify the prompt or input
5. Click **Run** to replay with the fix
6. Compare output side-by-side with the original

### Option 2: Programmatic Replay

```python
from langsmith import Client

client = Client()

def replay_failed_run(
    original_run_id: str,
    fixed_system_prompt: str,
) -> str:
    """
    Retrieve the input from a failed run and replay with a fixed system prompt.
    """
    # Get the original run
    run = client.read_run(original_run_id)
    original_inputs = run.inputs

    print(f"Original run: {run.name} | Status: {run.status}")
    print(f"Original input: {str(original_inputs)[:200]}")

    # Replay with the same input but a fixed prompt
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Extract the human message from the original inputs
    messages = original_inputs.get("messages", [])
    human_content = messages[-1]["content"] if messages else "Unknown input"

    response = llm.invoke(
        [
            SystemMessage(fixed_system_prompt),
            HumanMessage(human_content),
        ],
        config={
            "run_name": f"replay-of-{original_run_id[:8]}",
            "tags":     ["replay", "fix-verification"],
            "metadata": {"original_run_id": original_run_id},
        },
    )
    return response.content
```

---

## Dataset and Evaluator Workflow

When you identify a class of failures (not just one), the right fix is:

1. Create a dataset of representative inputs
2. Add your proposed fix
3. Run the evaluator to compare before/after

```python
from langsmith import Client

client = Client()

def add_failed_run_to_dataset(
    run_id: str,
    dataset_name: str,
    expected_output: str,
) -> None:
    """
    Add a failed run to a LangSmith dataset for regression testing.
    If the dataset doesn't exist, create it.
    """
    # Get or create dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Failed runs for regression testing",
        )

    # Read the original run's input
    run = client.read_run(run_id)
    inputs = run.inputs

    # Add as an example with the expected (correct) output
    client.create_example(
        inputs=inputs,
        outputs={"output": expected_output},
        dataset_id=dataset.id,
    )
    print(f"Added run {run_id[:8]} to dataset '{dataset_name}'")

# Later, run your evaluator against the dataset:
# client.run_on_dataset(
#     dataset_name="failed-runs-dataset",
#     llm_or_chain_factory=lambda: your_graph,
#     evaluation=evaluate.EvaluatorConfig(...)
# )
```

---

## Reading State at a Specific Checkpoint

For LangGraph agents with MemorySaver, you can inspect the state at any point in
history — not just the final state:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# Assuming graph is compiled with checkpointer
config = {"configurable": {"thread_id": "thread-001"}}

# Get the current (latest) state
current_state = graph.get_state(config)
print("Current state:", current_state.values)

# List all checkpoints (history of states)
for checkpoint_tuple in graph.get_state_history(config):
    print(
        f"Checkpoint at node: {checkpoint_tuple.metadata.get('source')}"
        f" | step: {checkpoint_tuple.metadata.get('step')}"
    )

# Get state at a specific checkpoint (time-travel debugging)
# Use the checkpoint_id from the history list:
# specific_state = graph.get_state(
#     config=config,
#     checkpoint_id="checkpoint-id-from-history"
# )
```

---

## Common Pitfalls

| Pitfall                                      | Symptom                                      | Fix                                                                           |
| -------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| No `run_name` set                            | Can't find the specific run to replay        | Always set `run_name` with meaningful context                                 |
| Replaying without tracking the replay        | Original and replay look identical in traces | Add `metadata.original_run_id` to the replay config                           |
| Only saving failed examples to dataset       | Dataset doesn't represent normal inputs      | Include both failure cases AND passing cases for balanced evaluation          |
| `get_state_history` on ephemeral MemorySaver | History lost when process restarts           | Use SqliteSaver or PostgresSaver for persistent history in debugging sessions |
| Comparing runs with different model versions | Diff is confounded by model change           | Fix the model; only change one variable at a time                             |

---

## Mini Summary

- The LangSmith trace tree shows latency, tokens, and cost per node — click the slowest span to identify bottlenecks
- "Open in Playground" lets you replay any span with a modified prompt — the fastest way to test a fix
- Programmatic replay captures the original input from the failed run and re-runs with a fix
- Add failing runs to a LangSmith dataset to build a regression test suite for future changes
- Use `graph.get_state_history()` with MemorySaver to inspect state at any checkpoint during debugging

---

[← Run Metadata](02-run-metadata-and-tags.md) | [Next → Evaluator Hooks](04-evaluator-hooks.md)
