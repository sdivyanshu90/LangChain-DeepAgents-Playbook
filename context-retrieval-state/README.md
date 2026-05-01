# Level 2: Context, Retrieval, and State

Level 2 is where LangChain applications stop being prompt pipelines and start behaving like information systems.

The central shift at this level is simple: useful AI applications rarely rely on the model alone. They need memory, external knowledge, retrieval strategy, and deliberate context management.

## Why This Level Matters

Most real LLM applications fail for predictable reasons:

- they forget prior context
- they inject too much irrelevant information
- they retrieve weak evidence
- they answer confidently when the source material is incomplete
- they have no stable model of session state or user context

This level teaches the design patterns that address those failures directly.

## Modules

- [Module 2.1: Memory fundamentals](modules/module-2.1-memory-fundamentals/README.md)
- [Module 2.2: RAG end-to-end](modules/module-2.2-rag-end-to-end/README.md)
- [Module 2.3: Retrieval quality](modules/module-2.3-retrieval-quality/README.md)
- [Module 2.4: Stateful applications](modules/module-2.4-stateful-applications/README.md)

## Projects

- [Project 2.1: DocuChat](projects/project-2.1-docuchat/README.md)
- [Project 2.2: Knowledge Base Q&A](projects/project-2.2-knowledge-base-qa/README.md)
- [Project 2.3: Support Bot](projects/project-2.3-support-bot/README.md)
- [Project 2.4: Meeting Notes Assistant](projects/project-2.4-meeting-notes-assistant/README.md)
- [Project 2.5: Research Digest Builder](projects/project-2.5-research-digest-builder/README.md)

## Recommended Order

1. Start with Module 2.1 to understand what memory is and what it is not.
2. Continue to Module 2.2 to build an end-to-end retrieval pipeline.
3. Study Module 2.3 before assuming retrieval quality is solved.
4. Read Module 2.4 to understand controlled state injection.
5. Build the projects in order, because each one introduces a slightly more realistic application surface.

## Shared Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r context-retrieval-state/requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## Learning Outcome

By the end of Level 2, you should be able to explain and implement:

- session memory and windowed conversation handling
- retrieval pipelines built from loaders, splitters, embeddings, vector stores, and retrievers
- citation-aware answer generation
- retrieval tuning decisions such as chunk size and overlap
- controlled state injection for user preferences and application context

This level is the bridge between simple chains and durable agent workflows. If Level 1 teaches composition, Level 2 teaches context discipline.
