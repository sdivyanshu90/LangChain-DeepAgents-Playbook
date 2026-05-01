# Project 1.2: Prompt Playground

Experiment with prompt templates, roles, and output modes using a reusable LCEL-based CLI.

## Learning Objective

Learn how prompt design changes model behavior, and how to package prompt experimentation into a reusable application rather than a notebook full of one-off tests.

## Real-World Use Case

Teams frequently need to compare different prompting strategies before they settle on a production prompt. A playground gives them a controlled way to test role instructions, tones, and output formats without rewriting the entire app each time.

## Difficulty

Beginner

## Skills Covered

- LCEL composition
- reusable prompt presets
- prompt variables and message roles
- text output versus structured output
- CLI ergonomics for experimentation

## Architecture Overview

The application separates concerns into four pieces:

1. Configuration for model settings.
2. Prompt presets that define the interaction style.
3. A chain builder that selects text or structured output mode.
4. A CLI that makes experimentation fast.

This matters because prompt engineering becomes much clearer when the prompt itself is treated as data, not buried inside a long function.

## Input and Output Expectations

Input:

- a task description
- a target audience
- a tone
- a prompt preset
- an output mode

Output:

- plain text for quick experiments
- or a structured JSON outline for more controlled comparison

## Dependencies and Setup

```bash
cd foundations/projects/project-1.2-prompt-playground
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-1.2-prompt-playground/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── prompt_playground/
│       ├── __init__.py
│       ├── chain.py
│       ├── cli.py
│       ├── config.py
│       ├── presets.py
│       └── schemas.py
└── tests/
    └── test_presets.py
```

## Step-by-Step Build Instructions

### 1. Define prompt presets

Each preset should capture a different reasoning or communication style.

### 2. Build one shared prompt shape

Use the preset plus task variables to keep the interface stable.

### 3. Support multiple output modes

This is the practical way to compare loose versus strict responses.

### 4. Expose everything through the CLI

If experimentation is annoying, people stop doing it and jump to production too early.

## Core Implementation Code

List presets:

```bash
PYTHONPATH=src python -m prompt_playground.cli --list-presets
```

Run a text experiment:

```bash
PYTHONPATH=src python -m prompt_playground.cli \
  --preset teacher \
  --task "Explain what a vector store is" \
  --audience "junior backend developers" \
  --tone "clear and concrete"
```

Run a structured experiment:

```bash
PYTHONPATH=src python -m prompt_playground.cli \
  --preset strategist \
  --task "Turn our support backlog themes into an action plan" \
  --audience "support leads" \
  --tone "concise" \
  --format json-outline
```

## Important Design Choices

### Why keep presets as explicit data?

Because prompt experimentation should be inspectable and versionable.

### Why include structured output in a playground?

Because real prompt work is not only about sounding better. It is also about producing data that the rest of the system can trust.

### Why use LCEL here?

Because this project is meant to teach reusable composition, not one-off scripting.

## Common Pitfalls

- changing too many prompt variables at once and learning nothing
- comparing prompts without controlling output format
- using a playground only for creativity, not application constraints
- burying presets inside condition-heavy procedural code

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then compare the same task across multiple presets and output modes. Good prompt evaluation begins with controlled input differences.

## Extension Ideas

- add saved prompt experiment sessions
- export comparisons to Markdown
- include automatic prompt version labels
- attach simple evaluation questions to each run

## Optional LangSmith Instrumentation

This project is a good candidate for LangSmith because prompt experiments become easier to compare when traces are preserved.

## Optional Deployment or Packaging Notes

This playground can evolve into an internal prompt lab UI or a lightweight service for product and ops teams.
