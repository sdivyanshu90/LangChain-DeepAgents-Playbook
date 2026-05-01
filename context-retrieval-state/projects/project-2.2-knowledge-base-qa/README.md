# Project 2.2: Knowledge Base Q&A

Build a searchable document Q&A system with grounded answers and citations.

## Learning Objective

Learn how to index a directory of documents, retrieve relevant context, and return structured answers that include explicit citations and gaps.

## Real-World Use Case

Internal docs, product guides, implementation notes, and runbooks are only useful if people can query them quickly and trust where the answers came from. This project builds a reusable knowledge base assistant for exactly that scenario.

## Difficulty

Intermediate

## Skills Covered

- directory-based document loading
- retrieval with source metadata
- citation-aware answer generation
- structured outputs for answer contracts
- simple Markdown rendering for human review

## Architecture Overview

The system works in four layers:

1. Load text and Markdown documents from a directory.
2. Split and index them for retrieval.
3. Retrieve the most relevant chunks for a question.
4. Ask the model for a structured answer with citations and known gaps.

Why this design matters:

- structured answers make downstream automation safer
- citation objects preserve traceability
- explicit gaps reduce false confidence

## Input and Output Expectations

Input:

- a directory of `.md`, `.txt`, or `.rst` files
- a natural-language question
- a session id for follow-up questions

Output:

- a structured answer object
- source citations
- optional Markdown rendering for humans

## Dependencies and Setup

```bash
cd context-retrieval-state/projects/project-2.2-knowledge-base-qa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-2.2-knowledge-base-qa/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── knowledge_base_qa/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── indexer.py
│       ├── loaders.py
│       ├── qa.py
│       ├── renderer.py
│       └── schemas.py
└── tests/
    └── test_renderer.py
```

## Step-by-Step Build Instructions

### 1. Load documents with source ids

This is what makes citations possible later.

### 2. Split and index the documents

Retrieval needs chunks that are small enough to be relevant but large enough to preserve meaning.

### 3. Define a structured answer schema

This keeps the output consistent even when the questions vary.

### 4. Render the answer for humans

Typed data is for the system. Markdown is for the reader.

## Core Implementation Code

```bash
PYTHONPATH=src python -m knowledge_base_qa.cli \
  --docs-dir docs \
  --question "How do we handle launch readiness reviews?" \
  --session launch-review \
  --format markdown
```

The `--session` flag keeps in-memory conversation history so you can ask follow-up questions such as "What about the second point?" within the same CLI session id.

## Important Design Choices

### Why use structured output here?

Because a knowledge assistant often feeds other systems, not just a chat box.

### Why include gaps?

Because good retrieval systems should expose what the evidence does not fully answer.

### Why add message history to a QA system?

Because follow-up questions often depend on the previous turn even when retrieval is still refreshed on every query.

### Why render Markdown separately?

Because presentation should be deterministic after the answer object has been validated.

## Common Pitfalls

- omitting source ids and then trying to reconstruct citations later
- mixing multiple output shapes in the same CLI code path
- over-trusting retrieved context when it is only partially relevant
- hiding uncertainty instead of returning explicit gaps

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the app on a small docs directory and confirm that the answer, citations, and Markdown rendering stay consistent for the same input.

## Extension Ideas

- add metadata-based filters for teams or product areas
- support HTML rendering in addition to Markdown
- store evaluation cases for common questions
- add reranking before answer generation

## Optional LangSmith Instrumentation

LangSmith is valuable here because citation quality and retrieval traces are both important to inspect.

## Optional Deployment or Packaging Notes

This project can become a docs portal backend, an internal support assistant, or a searchable engineering handbook service.
