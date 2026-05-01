# Project 3.6: Sales Intelligence Agent

Research a company, gather signals, and produce a structured sales brief with a LangGraph workflow.

## Learning Objective

Learn how to coordinate company lookup, signal gathering, and final brief generation in a graph that produces a structured research asset instead of a loose summary.

## Real-World Use Case

Sales teams often need a quick, grounded briefing before an account conversation. A useful system should gather company facts, recent signals, and likely talking points before drafting the final brief.

## Difficulty

Advanced

## Skills Covered

- company research workflow design
- structured sales brief outputs
- multi-step signal gathering
- graph-based orchestration
- observable research state

## Architecture Overview

The workflow has four stages:

1. Look up the company profile.
2. Gather recent business signals.
3. Draft the structured sales brief.
4. Return the final account research package.

Why this design matters:

- profile and signal gathering stay explicit
- the brief is built from visible state
- the output shape is consistent enough for downstream use

## Input and Output Expectations

Input:

- company name

Output:

- company profile
- recent signals
- structured sales brief

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.6-sales-intelligence-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.6-sales-intelligence-agent/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── sales_intelligence_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_signal_lookup.py
```

## Step-by-Step Build Instructions

### 1. Add a profile lookup tool

The workflow should start with the company's basic operating context.

### 2. Gather recent signals

Signals help turn a generic company summary into a usable sales brief.

### 3. Draft a structured brief

The final output should contain stable fields such as priorities, risks, and conversation hooks.

### 4. Keep the graph state inspectable

That makes it easier to see whether the brief is actually grounded in the gathered facts.

## Core Implementation Code

```bash
PYTHONPATH=src python -m sales_intelligence_agent.cli \
  --company "Northwind Robotics"
```

## Important Design Choices

### Why separate profile and signal gathering?

Because static company context and recent changes are different inputs with different value.

### Why use a structured brief?

Because sales intelligence often feeds CRM notes, prep docs, and downstream account workflows.

### Why use live Tavily search here?

Because sales research becomes more realistic when profile and news gathering reflect current external information instead of a tiny starter dataset.

## Common Pitfalls

- producing generic account summaries with no recent signals
- hiding the gathered evidence from the workflow state
- letting the final brief drift beyond the collected facts
- collapsing multiple research steps into one opaque prompt

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the workflow with different companies and confirm the resulting brief changes with the company profile and signal set.

## Extension Ideas

- add stakeholder-specific briefing modes
- add confidence and evidence scoring to each talking point
- add follow-up email draft generation

## Optional LangSmith Instrumentation

Tracing is useful here because company lookup, signal gathering, and brief drafting are distinct stages worth inspecting.

## Optional Deployment or Packaging Notes

This project can evolve into an account planning assistant, CRM copilot, or presales research tool.
