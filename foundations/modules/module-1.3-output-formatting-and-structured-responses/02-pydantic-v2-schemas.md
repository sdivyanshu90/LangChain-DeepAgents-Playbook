# 02 — Pydantic v2 Schemas

> **Previous:** [01 → Why Structured Output](01-why-structured-output.md) | **Next:** [03 → .with_structured_output()](03-with-structured-output.md)

---

## Real-World Analogy

A Pydantic model is like a customs declaration form.
The form has specific fields — name, date of birth, items declared, value in dollars.
If you write "lots of stuff" in the items box, the agent rejects the form.
The schema enforces that every field is present and in the right format
before anyone downstream processes the declaration.

Your LLM output is the declaration.
Pydantic is the customs agent.

---

## Why Pydantic v2, Not Dataclasses or TypedDict

You could use Python dataclasses or TypedDict to describe your expected structure.
But they give you no validation — they are just type hints.
Pydantic v2 actually enforces the types at runtime,
provides coercion (turning `"8"` into `8`),
and generates the JSON Schema that LangChain uses to instruct the model.

```
TypedDict:      type hints only, no runtime enforcement
dataclass:      type hints only, no runtime enforcement
Pydantic v2:    type hints + runtime validation + coercion + JSON Schema export
                ↑ this is what LangChain's structured output requires
```

---

## BaseModel — The Foundation

Every schema starts by subclassing `BaseModel`:

```python
from pydantic import BaseModel

# The simplest possible schema
class SentimentResult(BaseModel):
    sentiment: str    # model must produce a string for this field
    score: int        # model must produce a number (or a string that coerces to one)
```

That is all you need for simple cases.
But real production schemas need field descriptions, constraints, and validation.

---

## Field() — Adding Metadata and Constraints

`Field()` annotates each field with a description (which the model sees),
default values, and validation constraints.

```python
from pydantic import BaseModel, Field
from typing import Optional

class SentimentResult(BaseModel):
    sentiment: str = Field(
        description="The overall sentiment: 'positive', 'negative', or 'neutral'",
        # ↑ This description is passed to the model as context for this field.
        # The model reads it when deciding what value to fill in.
    )
    score: int = Field(
        description="Sentiment intensity score from 1 (very negative) to 10 (very positive)",
        ge=1,          # greater-than-or-equal: minimum allowed value
        le=10,         # less-than-or-equal: maximum allowed value
    )
    confidence: float = Field(
        description="Model's confidence in the result, 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: Optional[str] = Field(
        default=None,   # Optional field — may be absent without error
        description="Brief explanation of why this sentiment was assigned",
    )
```

### Field() Constraint Reference

| Argument          | Type     | Meaning                                                      |
| ----------------- | -------- | ------------------------------------------------------------ |
| `description`     | `str`    | Shown to the model; also appears in JSON Schema docs         |
| `default`         | any      | Value used when the field is absent                          |
| `default_factory` | callable | For mutable defaults like `list`, use `default_factory=list` |
| `ge`              | number   | Greater than or equal (≥)                                    |
| `gt`              | number   | Strictly greater than (>)                                    |
| `le`              | number   | Less than or equal (≤)                                       |
| `lt`              | number   | Strictly less than (<)                                       |
| `min_length`      | int      | Minimum string or list length                                |
| `max_length`      | int      | Maximum string or list length                                |
| `pattern`         | str      | Regex pattern the string must match                          |
| `examples`        | list     | Example values shown in JSON Schema                          |

---

## Enums — Restricting to a Fixed Set of Values

When a field must be one of a fixed set of values, use a Python `Enum`.
This is stronger than `description="one of: positive, negative, neutral"`
because Pydantic rejects any other value — the model cannot invent new categories.

```python
from enum import Enum
from pydantic import BaseModel, Field

class SentimentLabel(str, Enum):
    # Inheriting from str makes it JSON-serialisable without extra work
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"
    MIXED    = "mixed"   # catch-all for ambiguous reviews

class ReviewAnalysis(BaseModel):
    sentiment: SentimentLabel = Field(
        description="The overall tone of the review"
        # Pydantic will reject any value not in SentimentLabel
    )
    score: int = Field(ge=1, le=10, description="Score out of 10")
```

If the model returns `"very positive"`, Pydantic raises `ValidationError`.
That failure is caught and retried, not silently stored.

---

## Nested Models — Structured Hierarchies

Real extraction often has hierarchical structure.
Nest models to reflect that hierarchy:

```python
from typing import List
from pydantic import BaseModel, Field

class EntityMention(BaseModel):
    """A single named entity found in the text."""
    name: str = Field(description="The entity as it appears in the text")
    entity_type: str = Field(
        description="Type: PERSON, ORGANIZATION, LOCATION, PRODUCT"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident the extraction is, 0.0 to 1.0"
    )

class DocumentAnalysis(BaseModel):
    """Full analysis of a document."""
    summary: str = Field(
        max_length=500,
        description="One-paragraph summary of the document"
    )
    entities: List[EntityMention] = Field(
        default_factory=list,   # empty list if no entities found — not None
        description="All named entities mentioned in the document"
    )
    sentiment: SentimentLabel = Field(
        description="Overall document sentiment"
    )
    word_count_estimate: int = Field(
        description="Rough estimate of the document word count"
    )
```

Nesting works to any depth.
Keep nesting shallow (2-3 levels) — deeper hierarchies confuse models.

---

## field_validator — Custom Validation Logic

When built-in constraints are not enough, write a validator:

```python
from pydantic import BaseModel, Field, field_validator

class EmailExtraction(BaseModel):
    email: str = Field(description="The extracted email address")
    domain: str = Field(description="The domain part of the email (after @)")

    @field_validator("email")
    @classmethod
    def must_be_valid_email(cls, v: str) -> str:
        # v is the value the model produced for this field
        if "@" not in v:
            raise ValueError(f"'{v}' is not a valid email address")
        return v.lower()   # normalise to lowercase while we're here

    @field_validator("domain")
    @classmethod
    def must_match_email_domain(cls, v: str) -> str:
        # field_validator runs field by field — use model_validator for cross-field
        if "." not in v:
            raise ValueError(f"Domain '{v}' must contain at least one dot")
        return v.lower()
```

`field_validator` raises `ValueError` to signal invalid data.
Pydantic catches it and wraps it in `ValidationError` with a clear message.

---

## model_validator — Cross-Field Validation

When one field's validity depends on another field's value:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional

class DateRange(BaseModel):
    start_year: int = Field(description="The starting year of the range")
    end_year: Optional[int] = Field(
        default=None,
        description="The ending year, or null if ongoing"
    )

    @model_validator(mode="after")
    def end_must_be_after_start(self) -> "DateRange":
        # mode="after" runs after all individual fields are validated
        # self is the fully constructed model instance
        if self.end_year is not None and self.end_year < self.start_year:
            raise ValueError(
                f"end_year ({self.end_year}) must be >= start_year ({self.start_year})"
            )
        return self   # must return self in mode="after" validators
```

---

## Optional vs Union — The Type Annotation Difference

This is one of the most common sources of confusion.

```python
from typing import Optional, Union

class Example(BaseModel):
    # Optional[str] means: str or None
    # It is exactly equivalent to Union[str, None]
    # Use when the field genuinely may not exist in the source
    nickname: Optional[str] = None          # absent is fine, defaults to None

    # Union[str, int] means: either a string OR an integer
    # Use when the field is always present but can be different types
    identifier: Union[str, int]             # could be "abc123" or 42

    # list[str] vs Optional[list[str]]
    tags: list[str] = Field(default_factory=list)   # always a list, possibly empty
    primary_tag: Optional[str] = None               # may genuinely be absent
```

**Key rule:** `Optional` is for absence. `Union` is for type alternatives.
Do not make every field Optional "just in case" — it removes the contract guarantees.

---

## Viewing the Generated JSON Schema

You can see exactly what LangChain sends to the model:

```python
import json

# model_json_schema() returns the JSON Schema dict
schema = DocumentAnalysis.model_json_schema()
print(json.dumps(schema, indent=2))
# {
#   "title": "DocumentAnalysis",
#   "type": "object",
#   "properties": {
#     "summary": {
#       "type": "string",
#       "maxLength": 500,
#       "description": "One-paragraph summary of the document"
#     },
#     "entities": { ... },
#     ...
#   },
#   "required": ["summary", "sentiment", "word_count_estimate"]
# }
```

Fields with `default` or `default_factory` are NOT in the `required` list.
The model sees this schema when deciding what to output.

---

## Common Pitfalls

| Pitfall                                           | What goes wrong                                                           | Fix                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Using `Optional` on every field                   | No field is guaranteed; downstream code must handle `None` everywhere     | Only use `Optional` for genuinely optional fields           |
| Missing `description` on `Field()`                | Model does not know what the field means; produces garbage                | Add a clear `description` to every field                    |
| Using bare `list` instead of `list[str]`          | Pydantic can't validate element types                                     | Always parameterise generics: `list[str]`, `dict[str, int]` |
| Forgetting `default_factory=list` for list fields | First instance's list leaks to other instances (Pydantic raises an error) | Use `Field(default_factory=list)` for mutable defaults      |
| Deep nesting (5+ levels)                          | Model gets confused; inner fields hallucinated                            | Keep schemas flat; use 2 levels of nesting at most          |
| Enum values with spaces                           | JSON matching fails silently                                              | Use underscores: `"very_positive"` not `"very positive"`    |

---

## Mini Summary

- `BaseModel` is the foundation of every schema; subclass it for each output type.
- `Field()` adds descriptions (model reads them), defaults, and validation constraints.
- Enums enforce a closed set of values — stronger than a description saying "one of".
- Nested models reflect hierarchical data; keep nesting to 2 levels.
- `field_validator` validates a single field; `model_validator` handles cross-field rules.
- `Optional[X]` = `X or None` (use for absent fields). `Union[X, Y]` = X or Y type.
- Run `MyModel.model_json_schema()` to see exactly what the model receives.
