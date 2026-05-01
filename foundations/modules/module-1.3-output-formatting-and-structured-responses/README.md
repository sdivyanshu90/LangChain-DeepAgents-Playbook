# Module 1.3 — Structured Output and Schema Design

> **Track:** Foundations | **Prerequisite:** Module 1.2 LCEL and Runnables

---

## Mental Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Structured Output Pipeline                           │
│                                                                         │
│   Your Intent (Python)         Model Call              Your App        │
│                                                                         │
│   class ExtractResult(         ┌──────────┐            result =        │
│       BaseModel):              │          │            ExtractResult(   │
│     name: str         ──────►  │  LLM     │  ──────►     name="Alice", │
│     score: int                 │          │              score=9        │
│     tags: list[str]            └──────────┘            )               │
│                                                                         │
│   Schema = Contract            Tool Calling            Validated Object │
│   (you define)                 (happens internally)    (guaranteed)     │
└─────────────────────────────────────────────────────────────────────────┘
```

The schema is your **contract with the model**.
It tells LangChain what shape the output must take,
and Pydantic validates the result before it reaches your code.

---

## Topics

| #   | File                                                         | What you will learn                                                |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| 01  | [01-why-structured-output.md](01-why-structured-output.md)   | Why free-form text breaks software; the cost of silent failures    |
| 02  | [02-pydantic-v2-schemas.md](02-pydantic-v2-schemas.md)       | BaseModel, Field, validators, nested models                        |
| 03  | [03-with-structured-output.md](03-with-structured-output.md) | How `.with_structured_output()` works internally                   |
| 04  | [04-retry-and-fallback.md](04-retry-and-fallback.md)         | Handling parse failures gracefully with retry loops                |
| 05  | [05-schema-design-patterns.md](05-schema-design-patterns.md) | Good vs bad schemas; enums, confidence fields, extraction patterns |

---

## Data Flow at a Glance

```
User Input / Document Text
        │
        ▼
┌────────────────────┐
│ ChatPromptTemplate │  ← System prompt names the schema fields
└────────┬───────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  llm.with_structured_output(MySchema)   │
│                                          │
│  Internally:                             │
│  1. Serialise schema → tool definition   │
│  2. Pass to model as "function call"     │
│  3. Receive JSON arguments back          │
│  4. Validate with Pydantic               │
└────────┬─────────────────────────────────┘
         │
         ▼  (on success)              (on failure)
┌────────────────┐               ┌────────────────┐
│  MySchema      │               │  OutputFixing  │
│  instance      │               │  Parser retry  │
│  (validated)   │               │  (up to 3x)    │
└────────────────┘               └────────────────┘
```

---

## Key Packages

```bash
pip install langchain-core langchain-openai pydantic
```

Pydantic v2 is the default in current LangChain releases.
If you see `pydantic.v1`, you are on a legacy compatibility shim — upgrade.

---

## How to Work Through This Module

1. Read each topic file top to bottom before touching code.
2. Run the examples in the `examples/` directory.
3. Try breaking the schema intentionally — see how validation catches it.
4. Move to the next topic only when you understand the **why**, not just the **how**.

That is the problem with raw LLM output. A human may understand it, but your application often needs fields, lists, categories, and values that can be validated and stored. Structured response patterns solve that gap.

## Why Structured Output Matters

Free-form model output is useful for reading, but brittle for software.

If downstream code depends on a stable result, you need a contract.

That is why structured output matters:

- it reduces parsing ambiguity
- it gives your code predictable fields
- it makes validation possible
- it improves debugging because failures become local and explicit

For beginner apps, this is often the difference between a demo and a usable tool.

## Common Output Strategies

### 1. Plain Text

Best for:

- direct user-facing answers
- quick exploration
- low-structure content

Risk:

- hard for application code to trust

### 2. JSON-Like Prompting

Best for:

- simple experiments
- provider-agnostic prototypes

Risk:

- models may still return malformed JSON or drift from the requested format

### 3. Schema-Based Structured Output

Best for:

- application workflows
- automation
- storage and downstream processing

Why it is the preferred pattern here:

- your schema becomes the contract
- field descriptions improve extraction quality
- invalid responses fail loudly instead of silently polluting the app

## Internal Mechanics

With modern LangChain, one strong pattern is:

1. Define a Pydantic schema.
2. Ask the model for output that matches that schema.
3. Let the model wrapper and LangChain handle the structured response.
4. Use the validated object directly in your application.

This is cleaner than asking for raw JSON and manually parsing it after the fact.

## Example

See [examples/structured_output.py](examples/structured_output.py).

```python
class StudyCard(BaseModel):
    concept: str
    explanation: str
    practice_question: str

structured_model = model.with_structured_output(StudyCard)
chain = prompt | structured_model
```

That small change is architecturally important. The output is no longer just text. It is now typed application data.

## Code Walkthrough

### Schema First

The schema forces you to decide what the application actually needs.

That is a design improvement, not just a typing improvement.

### Field Descriptions

Descriptions help the model understand the intended meaning of each field.

### Validated Result

The returned object can be rendered, stored, or transformed without fragile regex or manual cleanup.

## Best Practices

- define the schema before writing the prompt
- keep fields concrete and application-relevant
- avoid overly broad field names like `details` or `misc`
- prefer nullable fields or empty lists over invented values
- render structured output into user-facing formats after validation

## Common Pitfalls

- asking for too many fields too early
- using vague schema names that confuse both humans and models
- treating structured output as perfectly reliable without good prompt grounding
- forcing the model to guess missing information instead of allowing empty values

## Mini Summary

Structured output is one of the most practical upgrades you can make to an LLM application.

It turns model responses into typed, inspectable data that application code can actually depend on.

## Optional Challenge

Take a plain-text output from Module 1.1 or Project 1.2 and redesign it into a structured schema. Then compare how much simpler the downstream code becomes.
