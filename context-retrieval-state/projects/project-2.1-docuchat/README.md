# Project 2.1: DocuChat

Chat with PDF documents using retrieval plus conversational history.

## Learning Objective

Learn how to combine PDF loading, chunking, embeddings, retrieval, and session memory into a document chat workflow that behaves more like a real application than a one-shot demo.

## Real-World Use Case

Teams often need to ask follow-up questions against onboarding guides, contracts, internal runbooks, or product documentation in PDF form. A useful assistant must not only retrieve the right page fragments, but also remember what the user already asked in the current session.

## Difficulty

Intermediate

## Skills Covered

- PDF loading with community loaders
- chunking and vector indexing
- retrieval-augmented chat
- windowed session history
- citation-style source display

## Architecture Overview

The application uses five steps:

1. Load PDF pages into `Document` objects.
2. Split pages into retrievable chunks.
3. Index the chunks with embeddings in a persistent Chroma vector store.
4. Retrieve relevant chunks for the current question.
5. Answer with both retrieved context and recent session history.

Why this design matters:

- retrieval keeps the answer grounded in the PDF
- session history preserves conversational continuity
- citations make debugging and trust easier

## Input and Output Expectations

Input:

- a PDF file path
- a question
- a session id

Output:

- a grounded answer
- cited PDF page references
- persisted chat history for that session
- a reusable vector index stored in `.chroma_docuchat`

## Dependencies and Setup

```bash
cd context-retrieval-state/projects/project-2.1-docuchat
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-2.1-docuchat/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── docuchat/
│       ├── __init__.py
│       ├── chat.py
│       ├── cli.py
│       ├── config.py
│       ├── indexer.py
│       ├── loaders.py
│       └── session.py
└── tests/
    └── test_session_store.py
```

## Step-by-Step Build Instructions

### 1. Load the PDF into documents

Start by preserving page metadata, because it will become part of your citations.

### 2. Split and index the pages

Whole pages are often too broad for good retrieval, so you split them into smaller chunks.

### 3. Add session storage

This is what turns a document query tool into a conversational workflow.

### 4. Build the answer chain

The answer prompt should use both recent history and retrieved document context.

### 5. Render citations clearly

Users should be able to see where the answer came from.

## Core Implementation Code

```bash
PYTHONPATH=src python -m docuchat.cli \
  --pdf handbook.pdf \
  --question "What are the rollout prerequisites?" \
  --session-id onboarding-demo \
  --rebuild
```

The Chroma index persists between runs. Use `--rebuild` when you want to delete the stored index and regenerate it from the current PDF.

## Important Design Choices

### Why keep session history outside the vector store?

Because conversation history and document retrieval solve different problems. One is continuity, the other is evidence lookup.

### Why keep a window on message history?

Because recent turns usually matter more than the full transcript, and prompt budgets are finite.

### Why show page references?

Because PDF chat without traceability quickly becomes untrustworthy.

### Why persist the vector index?

Because rebuilding embeddings on every run wastes time and hides the difference between indexing and querying.

## Common Pitfalls

- indexing whole pages without chunking
- replaying the full chat history forever
- losing page metadata during splitting
- assuming a follow-up question needs no retrieval refresh

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then test with a short PDF and a repeated session id. Ask a follow-up question and confirm that the application preserves context while still citing retrieved pages.

## Extension Ideas

- add answer streaming
- add a page preview mode for cited chunks
- summarize long session history automatically

## Optional LangSmith Instrumentation

This project benefits from tracing because retrieval and conversation state are both worth inspecting when a response goes wrong.

## Optional Deployment or Packaging Notes

DocuChat can be turned into a FastAPI backend, a Streamlit app, or an internal knowledge assistant for PDF-heavy workflows.
