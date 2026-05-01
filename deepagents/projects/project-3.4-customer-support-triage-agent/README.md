# Project 3.4: Customer Support Triage Agent

Route support issues, summarize context, and recommend next steps with an explicit graph workflow.

## Learning Objective

Learn how to build a triage workflow that classifies support issues, summarizes the case context, and recommends a next action for the correct queue.

## Real-World Use Case

Support teams need consistent intake triage so urgent issues are escalated quickly and routine issues reach the right queue. This project models that as an explicit, observable workflow.

## Difficulty

Advanced

## Skills Covered

- graph-based routing
- typed case classification
- summarization as a workflow step
- next-step recommendation generation
- explicit operational state

## Architecture Overview

The workflow has four stages:

1. Classify the issue by queue and urgency.
2. Build a concise support summary.
3. Recommend the next action.
4. Return a structured triage result.

Why this design matters:

- routing logic is explicit
- the case summary is inspectable before action recommendation
- the workflow can be extended with escalation or human review nodes later

## Input and Output Expectations

Input:

- customer issue title
- issue description

Output:

- assigned queue
- urgency level
- support summary
- recommended next action

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.4-customer-support-triage-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.4-customer-support-triage-agent/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── customer_support_triage_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_classification_logic.py
```

## Step-by-Step Build Instructions

### 1. Add a deterministic classifier

Queue assignment should be inspectable and easy to tune.

### 2. Summarize the case separately

That gives the workflow a reusable case representation.

### 3. Add the next-step recommender

This is where the final guidance becomes more contextual.

### 4. Keep the triage result structured

Support tooling often needs fields, not just prose.

## Core Implementation Code

```bash
PYTHONPATH=src python -m customer_support_triage_agent.cli \
  --title "Billing charge mismatch" \
  --description "The customer says this month's invoice includes duplicate usage charges."
```

## Important Design Choices

### Why keep queue assignment deterministic in the starter version?

Because support routing logic should be easy to inspect before it becomes more sophisticated.

### Why separate summarization from routing?

Because the summary can later feed both human agents and downstream automations.

### Why use a graph for this?

Because triage is already a workflow: classify, summarize, recommend.

## Common Pitfalls

- mixing classification and summarization into one opaque model step
- hiding urgency logic in prompt wording alone
- returning free-form text when queue systems need stable fields
- skipping the explicit next-step recommendation stage

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then test with billing, security, and product-bug issues to confirm the queue and urgency routing changes as expected.

## Extension Ideas

- add sentiment signals to urgency routing
- add a review queue for ambiguous cases
- attach canned macros for common issue types
- log triage confidence for audit analysis

## Optional LangSmith Instrumentation

Tracing is useful here because support routing mistakes are easier to diagnose when each node is visible.

## Optional Deployment or Packaging Notes

This project can evolve into a support desk intake service, ops routing helper, or CRM triage component.
