# Project 3.5: Codebase Explorer

Inspect a repository, summarize its structure, and answer high-level questions through a graph workflow.

## Learning Objective

Learn how to turn filesystem inspection into a structured exploration workflow that inventories files, selects relevant artifacts, and produces a concise architecture-oriented answer.

## Real-World Use Case

Engineers often need a fast orientation pass over a new repository. A codebase explorer can inspect file structure, pull key files, and return a guided summary without pretending it deeply understands every line.

## Difficulty

Advanced

## Skills Covered

- filesystem inspection as tool logic
- structured exploration state
- selective file reading
- architecture summarization
- graph-based workflow orchestration

## Architecture Overview

The workflow has four stages:

1. Inventory the repository files.
2. Select the most relevant artifacts.
3. Read the chosen files into state.
4. Synthesize an architecture answer.

Why this design matters:

- exploration steps are explicit
- file selection stays inspectable
- the final answer reflects the files the workflow actually touched

## Input and Output Expectations

Input:

- repository path
- exploration question

Output:

- file inventory summary
- selected file paths
- final architecture answer

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.5-codebase-explorer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.5-codebase-explorer/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── codebase_explorer/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_inventory_logic.py
```

## Step-by-Step Build Instructions

### 1. Inventory the repo

Start with filenames and shallow structure before reading content.

### 2. Select files intentionally

The workflow should choose files that are likely to answer the question.

### 3. Read only the selected files

Selective reading keeps the workflow focused and observable.

### 4. Synthesize the answer from those artifacts

This is what makes the final answer defensible.

## Core Implementation Code

```bash
PYTHONPATH=src python -m codebase_explorer.cli \
  --repo-path /path/to/repo \
  --question "What are the main entry points and modules?"
```

## Important Design Choices

### Why inventory before reading files?

Because exploration should start with the structure, not with arbitrary content reads.

### Why select only a few files?

Because focused exploration is easier to trace and debug than uncontrolled scanning.

### Why use a local filesystem tool pattern here?

Because codebase exploration is a strong example of tool-driven agentic work.

## Common Pitfalls

- trying to read the whole repository at once
- skipping the explicit file selection stage
- answering without exposing which files informed the result
- letting the synthesizer drift beyond the inspected artifacts

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the explorer on a small repo and confirm the selected files and final answer line up with the actual repository structure.

## Extension Ideas

- add symbol-level indexing
- add language-aware file selection
- attach code excerpts to the final answer
- add a review node that requests more files when evidence is thin

## Optional LangSmith Instrumentation

Tracing is useful here because the inventory, selection, and read steps are the real control surface of the workflow.

## Optional Deployment or Packaging Notes

This project can support onboarding, architecture reviews, and repository discovery tools.
