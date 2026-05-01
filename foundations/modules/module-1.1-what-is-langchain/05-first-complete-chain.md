# 05 — Your First Complete Chain

> **Previous:** [04 → Output Parsers](04-output-parsers.md) | **Next:** [Module 1.2 → LCEL and Runnables](../module-1.2-lcel-and-runnables/README.md)

---

## What We Are Building

A complete chain that takes a block of unstructured contact text and returns a validated `ContactCard` object —
demonstrating all four primitives together: **model → prompt → parser → chain**.

```
"Hi, I'm Alex Chen from Acme Corp. My email is alex@acme.com"
                          │
                          ▼
              ChatPromptTemplate.invoke()
                          │
          ┌───────────────┴───────────────┐
          │ SystemMessage (role + schema) │
          │ HumanMessage  (raw text)      │
          └───────────────┬───────────────┘
                          │ list[BaseMessage]
                          ▼
                     ChatOpenAI
                          │ AIMessage
                          ▼
              .with_structured_output()
                          │ ContactCard (Pydantic model)
                          ▼
         ContactCard(name='Alex Chen', email='alex@acme.com',
                     company='Acme Corp', phone=None)
```

---

## The Full Implementation

```python
# foundations/modules/module-1.1-what-is-langchain/examples/complete_chain.py
"""
Complete Module 1.1 Example
────────────────────────────
Demonstrates: ChatOpenAI + ChatPromptTemplate + .with_structured_output()
composed into a single LCEL chain with retry logic.

Run:  python examples/complete_chain.py
Env:  OPENAI_API_KEY in .env
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, EmailStr, Field

# ── 1. Environment Setup ──────────────────────────────────────────────────────
load_dotenv()  # reads OPENAI_API_KEY from .env in the project root

# ── 2. Output Schema ──────────────────────────────────────────────────────────
# Define what the chain should return.
# Field descriptions are read by the model — make them specific.

class ContactCard(BaseModel):
    """Structured contact information extracted from unstructured text."""

    name: str = Field(
        description="Full name of the person. Capitalise properly."
    )
    email: str = Field(
        description="Email address in lowercase. Null if not found."
    )
    company: str = Field(
        description="Company or organisation name."
    )
    phone: str | None = Field(
        None,
        description="Phone number in E.164 format (+1XXXXXXXXXX) if present, else null.",
    )
    confidence: float = Field(
        description="Your confidence in the extraction, from 0.0 (low) to 1.0 (high).",
        ge=0.0,
        le=1.0,
    )

# ── 3. Model ──────────────────────────────────────────────────────────────────
# temperature=0 for deterministic extraction.
# Always load the model name from env so tests can swap to a cheaper model.

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    max_tokens=512,
)

# ── 4. Structured Output ──────────────────────────────────────────────────────
# This wraps the model so it returns a ContactCard object, not raw text.
# Internally it injects the Pydantic schema as a tool definition.

structured_llm = llm.with_structured_output(ContactCard)

# ── 5. Prompt Template ────────────────────────────────────────────────────────
# SystemMessage sets the extraction role.
# HumanMessage injects the raw text as the {input} variable.

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a contact extraction assistant. "
        "Extract all contact information from the provided text. "
        "Return null for any field not present in the text. "
        "Never invent information.",
    ),
    ("human", "Extract contact info from this text:\n\n{input}"),
])

# ── 6. Chain Assembly ─────────────────────────────────────────────────────────
# The | operator pipes: prompt → structured_llm
# Data flow:
#   dict {"input": str}
#   → list[BaseMessage]     (prompt.invoke)
#   → ContactCard           (structured_llm.invoke)

chain = prompt | structured_llm

# ── 7. Run the Chain ──────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Hi, I'm Alex Chen from Acme Corp. My email is alex@acme.com and my cell is +1-555-123-4567.",
    "Sarah Connor — Resistance HQ, sarah.connor@resistance.org",
    "Please reach out to our founder: Marcus, marcus@tech.io",  # phone absent
]

for text in SAMPLE_TEXTS:
    result: ContactCard = chain.invoke({"input": text})

    print(f"\n{'─' * 60}")
    print(f"INPUT:      {text}")
    print(f"name:       {result.name}")
    print(f"email:      {result.email}")
    print(f"company:    {result.company}")
    print(f"phone:      {result.phone or '(not found)'}")
    print(f"confidence: {result.confidence:.2f}")
```

---

## Line-by-Line Walkthrough

### Step 2 — The Schema

```python
class ContactCard(BaseModel):
    phone: str | None = Field(None, ...)
    confidence: float = Field(..., ge=0.0, le=1.0)
```

- `str | None` with a default of `None` — the model can return null for missing fields instead of inventing data.
- `ge=0.0, le=1.0` — Pydantic rejects any confidence value outside the 0–1 range, even if the model hallucinates `1.5`.
- **The Field descriptions are not comments — the model reads them to understand what to extract.**

### Step 4 — `.with_structured_output()`

```python
structured_llm = llm.with_structured_output(ContactCard)
```

Internally, this serialises `ContactCard` as a JSON Schema and injects it into the model request as a tool definition. The model must call that tool with a conforming JSON body. LangChain deserialises the response into a `ContactCard` instance. You never see the raw JSON.

### Step 6 — Chain Assembly

```python
chain = prompt | structured_llm
```

The `|` operator creates a `RunnableSequence`. At invoke time:

1. `prompt.invoke({"input": text})` → `list[BaseMessage]`
2. `structured_llm.invoke(list[BaseMessage])` → `ContactCard`

No intermediate variables. No state. Each stage transforms and passes forward.

---

## Testing the Chain Without an API Key

For unit tests, you can mock the LLM:

```python
from langchain_core.runnables import RunnableLambda
from unittest.mock import patch

# Replace structured_llm with a deterministic mock
mock_result = ContactCard(
    name="Test User",
    email="test@example.com",
    company="Test Corp",
    phone=None,
    confidence=0.99,
)

test_chain = prompt | RunnableLambda(lambda _: mock_result)

result = test_chain.invoke({"input": "anything"})
assert result.name == "Test User"
```

This pattern — separating the chain structure from the model — is how the tests in every project in this repo are written.

---

## Extending the Chain

### Add retry on parse failure

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

# Fall back to PydanticOutputParser + retry wrapper for providers
# that don't support function calling natively
fixing_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=ContactCard),
    llm=llm,
    max_retries=2,
)
chain_with_retry = prompt_with_format_instructions | llm | fixing_parser
```

### Add streaming for UI display

```python
# Stream the raw text while the structured result is processed
for chunk in (prompt | llm).stream({"input": text}):
    print(chunk.content, end="", flush=True)

# Then get the structured result separately
result = chain.invoke({"input": text})
```

### Add batch processing for lists of contacts

```python
# Process multiple contacts in parallel (not sequentially)
inputs = [{"input": t} for t in SAMPLE_TEXTS]
results = chain.batch(inputs)   # parallel API calls

for r in results:
    print(f"{r.name} | {r.email}")
```

---

## Common Pitfalls in This Pattern

| Pitfall                                                   | What you see                               | Fix                                                                          |
| --------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------- |
| Forgetting `temperature=0` for extraction                 | Different field values on identical inputs | Always use `temperature=0` for structured extraction                         |
| Missing `load_dotenv()`                                   | `AuthenticationError`                      | Call `load_dotenv()` before any model initialization                         |
| Nullable fields without `= None` default                  | `ValidationError` when field absent        | `field: str \| None = Field(None, ...)`                                      |
| Giant extraction prompts with every field described twice | Slow, expensive, inconsistent              | Put field semantics in Pydantic `Field(description=...)` — not in the prompt |
| Using `chain.invoke()` in a loop for 100 items            | 100 sequential round-trips                 | Use `chain.batch()` — it runs in parallel                                    |

---

## Mini Summary

The complete first chain demonstrates:

1. **Schema design drives extraction quality** — Field descriptions are instructions.
2. **`.with_structured_output()` abstracts the JSON Schema / tool-call plumbing** — your code works with typed objects.
3. **`temperature=0` is non-negotiable for structured extraction** — any randomness corrupts validated fields.
4. **`chain = prompt | structured_llm` is the minimal, composable baseline** — every advanced pattern in this curriculum extends this structure.
5. **Mock the LLM in tests** — test the chain structure independently of API availability.

---

## What's Next

You now understand the four primitives.
The next module explains the `|` operator, streaming, batching, and the full Runnable interface that makes them composable.

**Next: [Module 1.2 → LCEL and Runnables](../module-1.2-lcel-and-runnables/README.md)**
