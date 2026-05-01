# Level 1: Foundations

Level 1 is where the repository stops being a roadmap and starts becoming a working learning system.

The purpose of this level is simple: build a correct mental model for modern LangChain before moving into retrieval, state, and agent orchestration. If these basics are weak, the later material will feel magical, brittle, and hard to debug.

## Why This Level Matters

Beginners often jump straight to agents and tools, then get stuck because they do not yet understand the data flow underneath the application.

This level fixes that by teaching:

- how prompts become messages
- how messages are sent to models
- how model output is parsed into usable values
- how LCEL turns these steps into readable, composable pipelines
- how structured outputs reduce ambiguity in downstream application logic

## Modules

- [Module 1.1: What is LangChain?](modules/module-1.1-what-is-langchain/README.md)
- [Module 1.2: LCEL and runnables](modules/module-1.2-lcel-and-runnables/README.md)
- [Module 1.3: Output formatting and structured response patterns](modules/module-1.3-output-formatting-and-structured-responses/README.md)

## Projects

- [Project 1.1: Smart Formatter CLI](projects/project-1.1-smart-formatter-cli/README.md)
- [Project 1.2: Prompt Playground](projects/project-1.2-prompt-playground/README.md)
- [Project 1.3: FAQ Generator](projects/project-1.3-faq-generator/README.md)

## Recommended Order

1. Read Module 1.1 and run the example.
2. Read Module 1.2 and trace how LCEL changes application composition.
3. Read Module 1.3 and compare free-form output with structured output.
4. Build Project 1.1 to learn strict JSON generation.
5. Build Project 1.2 to experiment with prompt design decisions.
6. Build Project 1.3 to combine prompting and structured outputs into a real deliverable.

## Shared Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r foundations/requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## Learning Outcome

By the end of Level 1, you should be able to explain and implement the core LangChain flow:

`input -> prompt -> model -> parser -> application output`

That single pipeline shape is the foundation for everything that follows, including RAG pipelines, stateful graphs, and DeepAgents.
