# Project 2.3: Support Bot

Answer internal policy questions with retrieval, confidence signals, and escalation guidance.

## Learning Objective

Learn how to build a retrieval-based support assistant that stays grounded in policy text, reports uncertainty clearly, and recommends escalation when the evidence is insufficient.

## Real-World Use Case

Internal support teams routinely answer questions about finance rules, security requirements, HR processes, and operational policies. A useful assistant must answer from policy evidence, not from generic model intuition.

## Difficulty

Intermediate

## Skills Covered

- policy document loading with metadata
- retrieval with optional department scoping
- structured support decisions
- escalation-aware response design
- confidence and missing-information reporting

## Architecture Overview

The bot works in four stages:

1. Load policy documents and infer department metadata.
2. Index them for retrieval.
3. Retrieve evidence for the current question.
4. Return a structured support decision with confidence, cited policies, and escalation guidance.

Why this design matters:

- policy assistants should be conservative by default
- structured decisions are easier to audit than free-form text
- escalation is a feature, not a failure

## Input and Output Expectations

Input:

- a directory of policy documents
- a support question
- optional department scope

Output:

- grounded answer
- cited policies
- confidence level
- escalation flag and reason when needed

## Dependencies and Setup

```bash
cd context-retrieval-state/projects/project-2.3-support-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-2.3-support-bot/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── support_bot/
│       ├── __init__.py
│       ├── bot.py
│       ├── cli.py
│       ├── config.py
│       ├── indexer.py
│       ├── loaders.py
│       └── schemas.py
└── tests/
    └── test_loaders.py
```

## Step-by-Step Build Instructions

### 1. Load policy documents with department metadata

This helps scope retrieval when the question is domain-specific.

### 2. Index the policy content

Chunking and embeddings make the policies searchable without hand-built keyword logic.

### 3. Define the support decision schema

The answer should include not just content, but also confidence and escalation state.

### 4. Keep the prompt conservative

A policy bot should prefer explicit uncertainty over confident guessing.

## Core Implementation Code

```bash
PYTHONPATH=src python -m support_bot.cli \
  --policies-dir policies \
  --question "Do conference trips need manager approval?" \
  --department finance
```

## Important Design Choices

### Why model escalation explicitly?

Because operational assistants need a safe fallback path, not just a best-effort answer.

### Why include confidence?

Because support workflows benefit from a visible signal about answer reliability.

### Why keep department metadata?

Because many policy questions are only relevant within a narrower scope than the full corpus.

## Common Pitfalls

- answering from general model knowledge instead of policy evidence
- forgetting to include a safe fallback when context is weak
- treating all policy docs as equally relevant without metadata scope
- hiding missing information inside vague prose

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then test with both answerable and unanswerable questions. A good support bot should escalate clearly when policy evidence is insufficient.

## Extension Ideas

- add policy version metadata
- route questions to specialist departments automatically
- store support decisions for audit review
- attach exact policy excerpts to escalation cases

## Optional LangSmith Instrumentation

Tracing is useful here because policy assistants need visible evidence paths and failure analysis.

## Optional Deployment or Packaging Notes

This project can support internal helpdesks, policy portals, and employee enablement tools.
