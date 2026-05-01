# Project 3.12: Workflow Recovery Agent

Demonstrate retries, fallbacks, and checkpoint-style state recording in a recoverable LangGraph workflow.

## Learning Objective

Learn how to model workflow recovery with retry counters, fallback paths, and explicit checkpoint notes so failures become visible and recoverable instead of silent.

## Real-World Use Case

Autonomous workflows fail for real reasons: transient API issues, weak evidence, and partial task completion. A recovery agent demonstrates how to handle those cases without collapsing into brittle behavior.

## Difficulty

Advanced

## Skills Covered

- retry logic in LangGraph
- fallback path design
- checkpoint-style state recording
- recoverable workflow control
- failure visibility and diagnostics

## Architecture Overview

The workflow has four stages:

1. Attempt the primary task.
2. Record failure context when needed.
3. Retry within a controlled limit.
4. Fall back to a safe degraded result when retries are exhausted.

Why this design matters:

- failures are explicit
- retries are bounded
- checkpoints preserve diagnostic context
- fallback output is safer than silent collapse

## Input and Output Expectations

Input:

- task description

Output:

- final workflow status
- checkpoint log
- recovered or fallback result

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.12-workflow-recovery-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.12-workflow-recovery-agent/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── workflow_recovery_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_retry_logic.py
```

## Step-by-Step Build Instructions

### 1. Put retry state in the graph state

That makes retry behavior observable.

### 2. Record checkpoint events on failure

The workflow should preserve why it retried.

### 3. Bound the retry loop

Recovery patterns are useful only when they eventually stop.

### 4. Add a fallback result

The system should degrade safely instead of crashing invisibly.

## Core Implementation Code

```bash
PYTHONPATH=src python -m workflow_recovery_agent.cli \
  --task "refresh deployment report"
```

## Important Design Choices

### Why record checkpoints in state?

Because recovery workflows are hard to debug if failure history disappears.

### Why use a bounded retry loop?

Because unbounded retries make workflows unpredictable and expensive.

### Why include fallback output?

Because graceful degradation is better than hidden failure.

## Common Pitfalls

- retrying forever
- retrying without recording what failed
- using fallback too early without enough recovery attempts
- hiding degraded-mode output from the caller

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the workflow and inspect the checkpoint log to confirm the retry path and fallback behavior are visible.

## Extension Ideas

- persist checkpoints to disk or a store
- add different fallback paths by failure type
- add human-in-the-loop approval before retry exhaustion
- attach structured failure metadata to each checkpoint

## Optional LangSmith Instrumentation

Tracing is especially useful here because retries and fallbacks are easiest to understand as a sequence of state transitions.

## Optional Deployment or Packaging Notes

This project can support robust job runners, recovery-aware pipelines, and durable agent orchestration systems.
