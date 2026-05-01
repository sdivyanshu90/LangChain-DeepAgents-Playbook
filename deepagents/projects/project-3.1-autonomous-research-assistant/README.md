# Project 3.1: Autonomous Research Assistant

Search, read, synthesize, and log actions in a small autonomous research workflow.

## Learning Objective

Learn how to build a LangGraph-based research workflow that plans a task, gathers evidence from tools, synthesizes findings, and keeps an explicit action trail.

## Real-World Use Case

Research tasks rarely end after one search. Analysts need a system that can look up sources, inspect them, capture findings, and return a concise brief with visible reasoning steps.

## Difficulty

Advanced

## Skills Covered

- tool-using graph workflows
- stateful evidence collection
- action logging
- final synthesis with a chat model
- explicit workflow design instead of hidden agent loops

## Architecture Overview

The workflow has four stages:

1. Create a simple plan from the research question.
2. Search a local research corpus for relevant source ids.
3. Read the selected sources into the shared state.
4. Synthesize a final brief from the gathered evidence.

Why this design matters:

- the action trail is explicit
- evidence is stored before synthesis
- the final answer is grounded in the sources the workflow actually touched

## Input and Output Expectations

Input:

- a research question

Output:

- a short research brief
- selected source ids
- action log entries

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.1-autonomous-research-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.1-autonomous-research-assistant/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── autonomous_research_assistant/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_search_logic.py
```

## Step-by-Step Build Instructions

### 1. Model the state

Keep the research question, plan, selected sources, evidence, and action log in one explicit state object.

### 2. Add a search step

Searching should produce source ids, not final answers.

### 3. Read the sources into state

The graph should preserve the evidence it actually used.

### 4. Synthesize only after evidence collection

This keeps the report closer to the workflow's observed facts.

## Core Implementation Code

```bash
PYTHONPATH=src python -m autonomous_research_assistant.cli \
  --question "What are the main risks in AI support automation?"
```

## Important Design Choices

### Why log actions in state?

Because research workflows are easier to debug when the action trail is part of the result.

### Why separate search and read?

Because source selection and evidence extraction are different decisions.

### Why keep the corpus local in the starter project?

Because it lets the workflow stay runnable while focusing on graph design rather than external infrastructure.

## Common Pitfalls

- synthesizing before enough evidence is gathered
- hiding the action trail from the final result
- letting search results go straight to output without reading them
- overcomplicating the planner in a starter implementation

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the workflow with questions that should hit different parts of the corpus and confirm the selected sources and action log change accordingly.

## Extension Ideas

- replace the local corpus with live search APIs
- add a reflection node that requests more evidence when the brief is weak
- persist action logs for later audit
- add confidence scoring to the final brief

## Optional LangSmith Instrumentation

This is a strong tracing candidate because the search, read, and synthesis steps should be easy to compare in one trace.

## Optional Deployment or Packaging Notes

This project can grow into a research copilot, analyst helper, or lightweight report-generation backend.
