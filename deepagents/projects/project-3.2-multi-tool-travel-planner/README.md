# Project 3.2: Multi-tool Travel Planner

Combine search, maps-style lookup, budget estimation, and itinerary drafting in a graph workflow.

## Learning Objective

Learn how to coordinate multiple typed tools inside a planning graph so the system can produce a grounded itinerary instead of a single generic travel answer.

## Real-World Use Case

Travel planning requires multiple data sources and tradeoffs: transport, lodging, timing, and budget. This project models that as an explicit workflow instead of a one-shot prompt.

## Difficulty

Advanced

## Skills Covered

- multi-tool orchestration
- intermediate state accumulation
- budget reasoning
- itinerary synthesis
- graph-based planning instead of ad hoc chaining

## Architecture Overview

The workflow has four stages:

1. Gather flight options.
2. Gather hotel options.
3. Estimate trip budget.
4. Draft a final itinerary.

Why this design matters:

- each tool contributes a specific slice of the final plan
- intermediate outputs remain inspectable
- the itinerary is built from grounded option data rather than invented details

## Input and Output Expectations

Input:

- origin city
- destination city
- number of nights
- budget cap

Output:

- travel options summary
- estimated budget
- final itinerary recommendation

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.2-multi-tool-travel-planner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.2-multi-tool-travel-planner/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── multi_tool_travel_planner/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_budget_logic.py
```

## Step-by-Step Build Instructions

### 1. Define small travel tools

Keep flights, hotels, and local transport estimates separate.

### 2. Store the intermediate options in state

That makes itinerary drafting easier to inspect and debug.

### 3. Add a budget node

Budget calculation should be explicit rather than hidden inside the final prompt.

### 4. Synthesize the itinerary at the end

Use the collected options and budget note to draft the recommendation.

## Core Implementation Code

```bash
PYTHONPATH=src python -m multi_tool_travel_planner.cli \
  --origin "Bengaluru" \
  --destination "Singapore" \
  --nights 3 \
  --budget 1200
```

## Important Design Choices

### Why separate budget estimation from itinerary drafting?

Because cost logic is deterministic and should not be buried in generative text.

### Why use mock tool data in the starter project?

Because the architectural lesson is tool orchestration, not API integration.

### Why keep the itinerary as the final step?

Because it should depend on the gathered facts, not precede them.

## Common Pitfalls

- trying to draft the itinerary before collecting options
- mixing deterministic math with the final narrative generation
- hiding tool decisions from the state
- forgetting to compare the plan against the budget cap

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the planner with both generous and constrained budgets and compare how the budget note changes the final itinerary.

## Extension Ideas

- add date-aware planning
- add weather or local transit tools
- add a route-review node that checks budget overflow
- stream itinerary revisions step by step

## Optional LangSmith Instrumentation

Tracing is useful here because each tool call contributes a visible piece of the final plan.

## Optional Deployment or Packaging Notes

This project can evolve into a concierge assistant, trip ops helper, or itinerary planning backend.
