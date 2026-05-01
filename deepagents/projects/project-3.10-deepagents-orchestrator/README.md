# Project 3.10: DeepAgents Orchestrator

Coordinate multiple specialized sub-agents in a LangGraph workflow.

## Learning Objective

Learn how to model multi-agent coordination as an explicit graph where specialist nodes contribute partial work and an orchestrator synthesizes the result.

## Real-World Use Case

Complex tasks often benefit from multiple specialist roles instead of one monolithic agent. A planner may define the work, a researcher may gather facts, an evaluator may challenge gaps, and a coordinator may combine the result.

## Difficulty

Advanced

## Skills Covered

- multi-agent coordination patterns
- specialist node design
- orchestration state management
- synthesis across agent outputs
- explicit delegation instead of hidden prompting

## Architecture Overview

The workflow has four stages:

1. Create a simple task plan.
2. Run a research specialist.
3. Run a review specialist.
4. Synthesize the final orchestration result.

Why this design matters:

- each specialist has a clear responsibility
- delegation is visible in the graph
- the coordinator can inspect all partial outputs before finishing

## Input and Output Expectations

Input:

- a complex task prompt

Output:

- plan
- specialist notes
- final orchestrated response

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.10-deepagents-orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.10-deepagents-orchestrator/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── deepagents_orchestrator/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_specialists.py
```

## Step-by-Step Build Instructions

### 1. Separate specialist responsibilities

Do not let all work collapse back into one giant node.

### 2. Keep partial outputs in state

The orchestrator should synthesize from visible contributions.

### 3. Add a coordinator step last

That is where the combined answer is produced.

### 4. Keep delegation explicit in the graph

This is what makes the workflow understandable.

## Core Implementation Code

```bash
PYTHONPATH=src python -m deepagents_orchestrator.cli \
  --task "Create a launch-readiness plan for a new support automation pilot"
```

## Important Design Choices

### Why use specialists at all?

Because different reasoning roles often have different goals and failure modes.

### Why keep the plan in state?

Because the specialists should operate against a shared task framing.

### Why use a coordinating synthesis node?

Because multi-agent work only becomes useful when the outputs are recombined coherently.

## Common Pitfalls

- giving all specialists overlapping responsibilities
- hiding delegation inside prompt text instead of graph structure
- failing to preserve partial outputs for inspection
- treating orchestration as just more looping

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the orchestrator on a task that needs both planning and critique, and confirm the final response reflects both specialist contributions.

## Extension Ideas

- add a third domain specialist
- add confidence scoring per specialist contribution
- route tasks dynamically to specialists
- add checkpointing between delegation stages

## Optional LangSmith Instrumentation

Tracing is especially useful here because orchestration is mainly about understanding which specialist did what.

## Optional Deployment or Packaging Notes

This project can support workflow hubs, analyst copilots, and multi-role planning systems.
