# Project 2.5: Research Digest Builder

Transform a set of source documents into structured summaries and an aggregated research digest.

## Learning Objective

Learn how to run a multi-document synthesis workflow that first produces grounded source summaries and then aggregates them into a concise digest for a target audience.

## Real-World Use Case

Research, competitive analysis, policy review, and market scanning often produce many source documents but no clean final brief. This project builds the pipeline that converts those source materials into a reusable digest.

## Difficulty

Intermediate

## Skills Covered

- source document ingestion
- structured per-document summarization
- multi-document aggregation
- digest rendering for human consumption
- separating source-level reasoning from cross-document synthesis

## Architecture Overview

The builder uses four phases:

1. Load source documents.
2. Generate a structured brief for each source.
3. Retrieve the most relevant briefs for the requested digest topic.
4. Aggregate the retrieved briefs into a digest with themes, actions, and open questions.

Why this design matters:

- source-level grounding reduces synthesis drift
- retrieval keeps the final synthesis focused instead of summarizing every brief indiscriminately
- the aggregation stage sees normalized inputs instead of raw, noisy text
- the final digest becomes easier to review and reuse

## Input and Output Expectations

Input:

- a directory of research documents
- a topic
- a target audience

Output:

- per-source structured summaries
- aggregated digest JSON
- Markdown research brief

## Dependencies and Setup

```bash
cd context-retrieval-state/projects/project-2.5-research-digest-builder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-2.5-research-digest-builder/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── research_digest_builder/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── digest.py
│       ├── loaders.py
│       ├── renderer.py
│       ├── schemas.py
│       └── summarizer.py
└── tests/
    └── test_renderer.py
```

## Step-by-Step Build Instructions

### 1. Summarize each source first

This prevents the final digest from being built on raw, inconsistent inputs.

### 2. Keep the source summary schema compact

The aggregation stage should receive clean, comparable building blocks.

### 3. Retrieve the most relevant briefs for the digest topic

This teaches the difference between summarizing everything and retrieve-then-synthesize.

### 4. Aggregate into a digest for a target audience

The same sources can produce different digests for executives, researchers, or operators.

### 4. Render the result separately

The final brief should be deterministic once the digest object is produced.

## Core Implementation Code

```bash
PYTHONPATH=src python -m research_digest_builder.cli \
  --docs-dir research \
  --topic "AI support automation" \
  --audience "operations leaders"
```

## Important Design Choices

### Why summarize sources before aggregation?

Because map-then-reduce style synthesis is easier to inspect and debug.

### Why retrieve briefs before the final digest?

Because even after source summarization, the synthesis stage should still focus on the briefs that matter most to the requested topic.

### Why keep the digest structured?

Because research briefs often need to feed other systems such as dashboards or publishing steps.

### Why include open questions?

Because good synthesis should expose uncertainty, not just compress information.

## Common Pitfalls

- asking the model to synthesize many raw documents in one step
- losing source identity during aggregation
- blending audience instructions too early into source summarization
- producing a digest that sounds polished but hides uncertainty

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then inspect both the per-source briefs and the final digest. The digest should clearly reflect the source summaries instead of introducing unrelated claims.

## Extension Ideas

- add citation snippets to the final brief
- support sector- or competitor-specific tags
- emit HTML or slide-friendly output
- add evaluation prompts for digest quality

## Optional LangSmith Instrumentation

This project is a strong tracing candidate because source summarization and final synthesis are separate stages worth inspecting independently.

## Optional Deployment or Packaging Notes

This project can support analyst workflows, competitive intelligence, and research ops pipelines.
