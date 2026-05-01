# Project 1.1: Smart Formatter CLI

Turn messy notes into strict JSON using modern LangChain structured outputs.

## Learning Objective

Learn how to transform unstructured text into validated application data using prompt design, a chat model, and a Pydantic schema.

## Real-World Use Case

Teams constantly receive rough input such as call notes, meeting scraps, support handoffs, and intake messages. Before that information becomes useful, someone has to normalize it into a consistent shape.

This project builds that normalization step.

## Difficulty

Beginner

## Skills Covered

- prompt templating
- structured outputs with Pydantic
- CLI design for LLM workflows
- environment-based configuration
- basic error handling and validation

## Architecture Overview

The application follows a simple but production-relevant flow:

1. Read messy text from a file or standard input.
2. Build a prompt that asks for a normalized representation.
3. Call the model with `with_structured_output`.
4. Validate the result against a schema.
5. Print or save strict JSON.

Why this design matters:

- the prompt stays isolated from I/O code
- validation happens in one place
- downstream code gets typed data instead of brittle string parsing

## Input and Output Expectations

Input:

- raw free-form notes from a file or piped input

Output:

- JSON with topic, summary, key points, action items, risks, and follow-up questions

## Dependencies and Setup

```bash
cd foundations/projects/project-1.1-smart-formatter-cli
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`.

## File Structure

```text
project-1.1-smart-formatter-cli/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── smart_formatter/
│       ├── __init__.py
│       ├── chain.py
│       ├── cli.py
│       ├── config.py
│       └── schemas.py
└── tests/
    └── test_schemas.py
```

## Step-by-Step Build Instructions

### 1. Define the schema first

This is the key beginner habit. Decide what the application needs before asking the model for it.

### 2. Write the prompt around that schema

The prompt should explain what to extract and what not to invent.

### 3. Keep model configuration in one place

That makes debugging and later provider changes easier.

### 4. Build a CLI around the chain

This turns the example into a tool you can actually use.

### 5. Validate the output

Pydantic gives you a typed contract between the model and the rest of the app.

## Core Implementation Code

Run the project like this:

```bash
PYTHONPATH=src python -m smart_formatter.cli --input-file sample_note.txt --pretty
```

Or pipe data directly:

```bash
echo "Met with Alex from finance. Need updated budget by Friday." | PYTHONPATH=src python -m smart_formatter.cli --pretty
```

## Important Design Choices

### Why use structured output instead of free-form text?

Because application code needs stable fields, not paragraphs that have to be re-parsed later.

### Why keep the CLI separate from the chain?

Because interface code changes for operational reasons, while chain logic changes for model behavior reasons. Keeping them separate reduces coupling.

### Why require evidence-backed extraction?

Because beginners often let the model infer too much. Asking it to stay grounded reduces hallucinated structure.

## Common Pitfalls

- allowing empty input to reach the model
- assuming every note contains action items or deadlines
- forgetting that strict schemas still depend on prompt quality
- mixing file handling and prompt logic in the same function

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then run the CLI with short notes and confirm the output shape is stable. Finally, test edge cases such as notes with no owners, no dates, or incomplete context.

## Extension Ideas

- add optional CSV export
- support multiple output schemas for different teams
- include source evidence snippets for each extracted field
- add a batch mode that reads a directory of notes

## Optional LangSmith Instrumentation

Add LangSmith later by setting tracing environment variables and enabling tracing before invocation. For this beginner project, keep the first version simple.

## Optional Deployment or Packaging Notes

This project can be packaged as an internal CLI, a small FastAPI endpoint, or a background document-normalization step in a larger workflow.
