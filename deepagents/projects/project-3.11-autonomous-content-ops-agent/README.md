# Project 3.11: Autonomous Content Ops Agent

Gather sources, draft content, and self-check the result in a multi-step workflow.

## Learning Objective

Learn how to model a content operations workflow where the agent gathers source material, drafts content, reviews its own draft, and produces a revised output.

## Real-World Use Case

Content workflows often need more than text generation. They need source gathering, editorial framing, and self-review before the output is ready for a human editor.

## Difficulty

Advanced

## Skills Covered

- source gathering workflows
- content drafting from collected evidence
- self-check and revision patterns
- graph-based editorial control
- explicit intermediate artifacts

## Architecture Overview

The workflow has four stages:

1. Gather relevant source snippets.
2. Draft content from those snippets.
3. Run a self-check review.
4. Revise the draft based on review findings.

Why this design matters:

- sources are preserved before drafting
- review happens as a distinct node
- revision is grounded in explicit critique rather than vague iteration

## Input and Output Expectations

Input:

- content topic
- target audience

Output:

- gathered source notes
- initial draft
- review notes
- revised draft

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.11-autonomous-content-ops-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.11-autonomous-content-ops-agent/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── autonomous_content_ops_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_source_gathering.py
```

## Step-by-Step Build Instructions

### 1. Gather source notes first

Drafting should start from evidence, not from a blank prompt.

### 2. Keep draft and review separate

That gives the workflow a clearer self-check loop.

### 3. Revise from explicit review notes

The revision step should have concrete guidance, not only the original task.

### 4. Preserve all intermediate artifacts

This makes the content workflow easier to inspect and improve.

## Core Implementation Code

```bash
PYTHONPATH=src python -m autonomous_content_ops_agent.cli \
  --topic "Why AI support workflows need escalation paths" \
  --audience "operations managers"
```

## Important Design Choices

### Why gather sources even in a starter project?

Because content quality depends heavily on the inputs feeding the draft.

### Why add self-checking explicitly?

Because review and revision are part of the actual workflow, not optional polish.

### Why preserve the draft and revised draft separately?

Because iteration quality is hard to inspect if only the final output survives.

## Common Pitfalls

- drafting before enough source material is gathered
- treating review as vague style feedback instead of actionable critique
- discarding intermediate artifacts
- revising without grounding the revision in review findings

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the workflow on two topics and confirm the gathered sources and revised draft change with the topic and audience.

## Extension Ideas

- add source quality scoring
- add tone presets for different channels
- add a compliance review node after revision
- support publishing-ready formats such as Markdown and HTML

## Optional LangSmith Instrumentation

Tracing is useful here because source gathering, drafting, review, and revision are distinct content workflow stages.

## Optional Deployment or Packaging Notes

This project can support content teams, enablement operations, and internal publishing pipelines.
