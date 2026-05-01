# Project 3.7: Meeting-to-Action Agent

Extract decisions, tasks, owners, and deadlines from meeting transcripts using a graph workflow.

## Learning Objective

Learn how to convert raw meeting text into a structured action package through an explicit extraction workflow instead of a single opaque prompt.

## Real-World Use Case

Meeting transcripts often contain decisions and actions that disappear into long notes. A useful agent should extract the actionable artifacts and present them in a stable format for teams and systems.

## Difficulty

Advanced

## Skills Covered

- structured extraction with LLMs
- graph-based workflow design
- typed action records
- deterministic Markdown rendering
- action-oriented output design

## Architecture Overview

The workflow has three stages:

1. Normalize the meeting text.
2. Extract a structured action package.
3. Render the final action summary.

Why this design matters:

- extraction and rendering are separated
- the structured action package can feed other systems
- the final output is easier to audit than free-form prose

## Input and Output Expectations

Input:

- raw meeting transcript or notes

Output:

- summary
- decisions
- action items with owners and deadlines
- open questions

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.7-meeting-to-action-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.7-meeting-to-action-agent/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── meeting_to_action_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_renderer_logic.py
```

## Step-by-Step Build Instructions

### 1. Normalize the input text

This keeps the extraction step simpler and more stable.

### 2. Define the action schema first

The extraction target should be explicit before the model runs.

### 3. Run extraction as a dedicated node

This makes the workflow easier to test and extend.

### 4. Render the output after validation

That keeps presentation deterministic.

## Core Implementation Code

```bash
PYTHONPATH=src python -m meeting_to_action_agent.cli \
  --text "Alex decided we will delay launch by one week. Priya will update the rollout plan by Friday."
```

## Important Design Choices

### Why extract structured data before rendering?

Because the meeting output should be usable by systems, not just readable by humans.

### Why keep open questions separate from decisions?

Because unresolved items are operationally different from commitments.

### Why use a graph for this instead of one prompt?

Because extraction pipelines usually benefit from explicit preprocessing and output handling.

## Common Pitfalls

- mixing decisions and open questions into one field
- allowing the model to invent owners or deadlines
- tying rendering too tightly to the extraction prompt
- skipping schema design before implementation

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then test with notes that contain both explicit actions and unresolved questions to confirm the workflow separates them properly.

## Extension Ideas

- add attendee extraction
- add calendar-ready export formats
- support confidence flags per action item
- attach source snippets for every extracted decision

## Optional LangSmith Instrumentation

Tracing is useful here because extraction quality and rendering quality can be inspected separately.

## Optional Deployment or Packaging Notes

This project can support meeting ops, PM workflows, and action tracking backends.
