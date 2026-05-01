# Project 1.3: FAQ Generator

Generate clean FAQ content from raw notes and export both JSON and Markdown.

## Learning Objective

Combine prompt design, structured output, and rendering logic into a useful content-generation workflow.

## Real-World Use Case

Product notes, support logs, onboarding docs, and internal research often exist as rough text long before they become polished FAQ material. This project automates the first high-quality draft while keeping the output structured enough for review and reuse.

## Difficulty

Beginner

## Skills Covered

- schema-first design
- structured generation with LangChain
- rendering typed data into Markdown
- CLI-based workflow automation
- separating generation logic from presentation logic

## Architecture Overview

The system has four layers:

1. Input handling for source notes.
2. A structured generation chain that produces FAQ data.
3. A renderer that converts validated FAQ objects into Markdown.
4. A CLI that coordinates reading, generation, and export.

Why this design matters:

- the model generates data, not final formatting
- Markdown rendering becomes deterministic
- future delivery targets like HTML or CMS payloads become easier to add

## Input and Output Expectations

Input:

- source notes from a file or standard input
- audience and title context

Output:

- structured FAQ JSON
- Markdown FAQ ready for docs or README use

## Dependencies and Setup

```bash
cd foundations/projects/project-1.3-faq-generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-1.3-faq-generator/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── faq_generator/
│       ├── __init__.py
│       ├── chain.py
│       ├── cli.py
│       ├── config.py
│       ├── renderer.py
│       └── schemas.py
└── tests/
    └── test_renderer.py
```

## Step-by-Step Build Instructions

### 1. Define the FAQ schema

Decide what makes an FAQ entry valid before you ask the model to produce one.

### 2. Add audience context

Good FAQs are written for someone. That context should be explicit.

### 3. Generate structured data first

This prevents formatting concerns from being mixed into extraction and synthesis.

### 4. Render Markdown afterward

Presentation should be deterministic once the data is validated.

## Core Implementation Code

Generate an FAQ from a source file and print Markdown to stdout:

```bash
PYTHONPATH=src python -m faq_generator.cli \
  --source-file notes.txt \
  --title "Billing Migration FAQ" \
  --audience "customer success managers"
```

Write both JSON and Markdown artifacts:

```bash
PYTHONPATH=src python -m faq_generator.cli \
  --source-file notes.txt \
  --title "Billing Migration FAQ" \
  --audience "customer success managers" \
  --json-output faq.json \
  --markdown-output FAQ.md
```

## Important Design Choices

### Why generate JSON first?

Because the FAQ is application data before it is presentation content.

### Why separate rendering from generation?

Because it makes the output layer deterministic and testable.

### Why include tags and audience?

Because FAQ content is more reusable when it carries retrieval and publishing context.

## Common Pitfalls

- generating long marketing copy instead of actual FAQs
- skipping audience context and getting generic answers
- blending formatting instructions too heavily into the generation prompt
- forgetting to constrain the number of FAQ items

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then inspect both the JSON and Markdown outputs. The JSON should be schema-stable, and the Markdown should remain deterministic for the same structured input.

## Extension Ideas

- add citation snippets from the source notes
- group FAQs by topic sections
- export HTML alongside Markdown
- add a review mode that flags uncertain answers

## Optional LangSmith Instrumentation

This project benefits from LangSmith when you want to compare prompt revisions for FAQ quality across different note sets.

## Optional Deployment or Packaging Notes

This project can serve as a docs preprocessor, a support enablement helper, or a content-generation step inside a knowledge management pipeline.
