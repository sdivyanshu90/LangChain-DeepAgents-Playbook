# Project 3.8: Data Query Agent

Translate natural language questions into safe data access workflows using a guarded SQL execution path.

## Learning Objective

Learn how to structure a natural-language data access workflow that plans a query, validates it for safety, executes it against SQLite, and summarizes the results.

## Real-World Use Case

Business users often ask data questions in natural language, but a production system cannot let an LLM run arbitrary SQL. This project demonstrates a safer pattern: plan, validate, execute, summarize.

## Difficulty

Advanced

## Skills Covered

- safe workflow design for data access
- SQL guardrails
- graph-based validation and execution
- result summarization
- SQLite-backed agent tools

## Architecture Overview

The workflow has four stages:

1. Build a query plan from the question.
2. Generate SQL.
3. Validate the SQL against safety rules.
4. Execute and summarize the results.

Why this design matters:

- validation is a first-class node
- execution happens only after explicit safety checks
- the final answer stays grounded in actual query results

## Input and Output Expectations

Input:

- SQLite database path
- natural-language question

Output:

- generated SQL
- query rows
- summarized answer

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.8-data-query-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.8-data-query-agent/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── data_query_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_sql_validation.py
```

## Step-by-Step Build Instructions

### 1. Separate planning from execution

This makes the workflow easier to constrain.

### 2. Add SQL validation rules

The workflow should reject non-SELECT queries and obvious unsafe patterns.

### 3. Execute only after validation passes

This is the key safety boundary.

### 4. Summarize real query results

The final response should describe what the database returned, not what the model guessed.

## Core Implementation Code

```bash
PYTHONPATH=src python -m data_query_agent.cli \
  --db-path analytics.db \
  --question "How many open tickets belong to the billing team?"
```

## Important Design Choices

### Why gate execution behind validation?

Because safe data access is the whole point of the architecture.

### Why keep SQLite in the starter version?

Because it makes the workflow runnable while teaching the real control pattern.

### Why summarize rows in a final step?

Because users want answers, but those answers should remain grounded in the returned rows.

## Common Pitfalls

- executing generated SQL without inspection
- allowing non-SELECT statements through the workflow
- hiding validation failures instead of surfacing them clearly
- generating answers that are not tied to the row results

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then test with a small SQLite database and both safe and unsafe prompts to confirm validation blocks the unsafe path.

## Extension Ideas

- add schema introspection tooling
- support approved views instead of raw tables
- add row count limits automatically
- add human approval for high-risk query classes

## Optional LangSmith Instrumentation

Tracing is useful here because query planning, validation, and execution are separate decision points.

## Optional Deployment or Packaging Notes

This project can support analytics copilots, internal ops dashboards, and safe NL-to-SQL services.
