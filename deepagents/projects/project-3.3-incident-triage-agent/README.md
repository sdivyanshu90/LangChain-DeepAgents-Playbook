# Project 3.3: Incident Triage Agent

Analyze alerts, gather context, and propose operational next steps in a graph workflow.

## Learning Objective

Learn how to structure an incident triage workflow with contextual lookup, severity assessment, runbook retrieval, and a final recommendation step.

## Real-World Use Case

On-call teams need fast, grounded triage when alerts fire. A useful agent must gather the relevant alert context, map severity, consult runbooks, and propose actions without pretending to have certainty it does not have.

## Difficulty

Advanced

## Skills Covered

- graph-based incident handling
- typed severity decisions
- runbook-aware recommendations
- explicit operational state
- safe triage-oriented output design

## Architecture Overview

The workflow has four stages:

1. Parse the alert payload.
2. Gather contextual telemetry hints.
3. Retrieve a matching runbook.
4. Produce a structured triage recommendation.

Why this design matters:

- incident context and runbooks remain visible inputs
- severity is assessed explicitly
- recommendations are grounded in operational artifacts rather than generic prose

## Input and Output Expectations

Input:

- an alert title
- a short alert summary

Output:

- severity level
- likely cause summary
- recommended next actions
- cited runbook name

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.3-incident-triage-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.3-incident-triage-agent/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── incident_triage_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_severity_logic.py
```

## Step-by-Step Build Instructions

### 1. Normalize the alert input

Give the workflow a stable alert title and summary to work from.

### 2. Add severity assessment

This should be deterministic enough to inspect and improve.

### 3. Retrieve a runbook

Triage should anchor on the nearest operational playbook.

### 4. Synthesize the recommendation last

The final response should reflect the gathered context, not replace it.

## Core Implementation Code

```bash
PYTHONPATH=src python -m incident_triage_agent.cli \
  --title "API error rate spike" \
  --summary "5xx rate doubled in the last 10 minutes and checkout is timing out"
```

## Important Design Choices

### Why keep severity as explicit state?

Because routing, escalation, and response urgency often depend on it.

### Why consult a runbook before final output?

Because incident workflows should connect to operational practice, not just model intuition.

### Why use structured results?

Because triage output often feeds human operators or incident systems that expect stable fields.

## Common Pitfalls

- treating all alerts as equally severe
- returning generic advice without runbook grounding
- hiding missing context behind confident language
- skipping the explicit context-gathering step

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then test with both low-urgency and high-urgency alerts to confirm severity and runbook selection change sensibly.

## Extension Ideas

- add alert history lookup
- add service ownership routing
- add a verification node for noisy alerts
- add retry logic for transient telemetry lookup failure

## Optional LangSmith Instrumentation

Tracing is useful here because incident routing and runbook selection are both worth inspecting node by node.

## Optional Deployment or Packaging Notes

This project can evolve into an on-call assistant, incident desk helper, or SOC workflow component.
