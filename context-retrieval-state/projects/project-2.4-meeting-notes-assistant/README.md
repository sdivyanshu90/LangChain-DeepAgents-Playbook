# Project 2.4: Meeting Notes Assistant

Ingest meeting notes, summarize them into structured records, and retrieve decisions later.

## Learning Objective

Learn how to transform raw meeting notes into structured summaries, index those summaries for retrieval, and answer follow-up questions about decisions and action items.

## Real-World Use Case

Teams constantly lose operational memory inside long notes and scattered meeting recaps. This project builds a small system that turns raw notes into reusable decision records instead of leaving them as unstructured text.

## Difficulty

Intermediate

## Skills Covered

- note ingestion from files
- structured summarization
- indexing summary records for retrieval
- decision-focused retrieval
- deterministic Markdown rendering

## Architecture Overview

The assistant uses four stages:

1. Load note documents from a directory.
2. Summarize each note into a structured meeting record.
3. Index the summaries for retrieval.
4. Answer follow-up questions using the summary corpus.

Why this design matters:

- summarization converts noisy notes into stable application data
- retrieval works better on summaries when the user asks about decisions or owners
- rendering keeps the results easy to publish and review

## Input and Output Expectations

Input:

- a directory of meeting notes
- optional question about prior decisions or action items

Output:

- structured meeting summaries
- Markdown meeting digest
- grounded follow-up answers when a question is supplied

## Dependencies and Setup

```bash
cd context-retrieval-state/projects/project-2.4-meeting-notes-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-2.4-meeting-notes-assistant/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── meeting_notes_assistant/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── indexer.py
│       ├── loaders.py
│       ├── qa.py
│       ├── renderer.py
│       ├── schemas.py
│       └── summarizer.py
└── tests/
    └── test_renderer.py
```

## Step-by-Step Build Instructions

### 1. Load notes with source metadata

File names often become the first useful identifiers when users ask follow-up questions.

### 2. Summarize each note into a schema

This turns meeting notes into reusable records rather than raw prose blobs.

### 3. Index the summaries

Users usually ask for decisions, actions, or owners, not the full original note.

### 4. Answer targeted follow-up questions

The retrieval layer should search the summaries, not only the raw notes.

## Core Implementation Code

```bash
PYTHONPATH=src python -m meeting_notes_assistant.cli \
  --notes-dir notes \
  --question "What decisions were made about launch timing?"
```

## Important Design Choices

### Why summarize before retrieval?

Because user questions about meetings usually target distilled facts such as decisions and action items.

### Why keep a Markdown renderer?

Because summaries are often read by humans before they are queried by systems.

### Why separate ingestion and querying logic?

Because meeting summarization and decision retrieval are related but distinct behaviors.

## Common Pitfalls

- treating long raw notes as the only retrieval surface
- skipping source metadata on summaries
- letting the model invent decisions that were not in the notes
- forgetting that many notes contain unresolved questions, not just conclusions

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the assistant on a small note set and inspect both the summary output and a few retrieval queries about decisions and action items.

## Extension Ideas

- persist structured summaries to a JSONL store
- add per-meeting attendee extraction
- index raw notes and summaries together for hybrid retrieval
- add uncertainty flags for low-confidence summaries

## Optional LangSmith Instrumentation

Tracing is useful here because summarization quality and retrieval quality both affect the final answer.

## Optional Deployment or Packaging Notes

This project can support team ops, project reviews, and internal meeting archives.
