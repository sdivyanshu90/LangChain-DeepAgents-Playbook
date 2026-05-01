# 04 — Output Parsers

> **Previous:** [03 → Prompts and Messages](03-prompts-and-messages.md) | **Next:** [05 → Your First Complete Chain](05-first-complete-chain.md)

---

## Real-World Analogy

A model without a parser is like receiving a handwritten letter when you needed a spreadsheet.
The information is there, but your software cannot use it.

An output parser is the translation layer between what the model produces (text)
and what your application expects (a typed, validated data structure).

---

## Why Parsing Is Not Optional

```
Model output (raw):
  "The capital is Paris. Population approximately 2.1 million. Currency: Euro."

What your application actually needs:
  {"city": "Paris", "population": 2100000, "currency": "EUR"}
```

Without a parser, every downstream function manually slices and dices strings.
That is fragile, inconsistent, and untestable.

A parser makes output failure **explicit and local** instead of hidden and downstream.

---

## The Three Main Parsers

```
┌─────────────────────────────────────────────────────────────┐
│                    Output Parser Spectrum                     │
│                                                               │
│  StrOutputParser          JsonOutputParser   PydanticParser   │
│  ───────────────         ────────────────   ──────────────   │
│  text → str              text → dict        text → Model      │
│                                                               │
│  ◄── simpler ──────────────────────────────── stronger ──►   │
│  ◄── less validation ──────────────── more validation ──►    │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. StrOutputParser

The simplest possible parser. Extracts `.content` from an `AIMessage` and returns a plain string.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Without LCEL:
ai_message = llm.invoke([HumanMessage("What is 2+2?")])
result = parser.invoke(ai_message)   # "4"

# With LCEL (the standard way):
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is 2+2?"})  # "4"
```

**When to use it:**

- User-facing responses that don't need post-processing
- Summaries, explanations, chat answers
- Any time downstream code just needs a string

**What it does NOT do:**

- Validate format
- Parse structure
- Ensure completeness

---

## 2. JsonOutputParser

Instructs the model to produce JSON and parses the response into a Python dict.

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

template = ChatPromptTemplate.from_messages([
    ("system", "Return valid JSON only. No explanation text outside the JSON."),
    ("human",  "Extract: name, email, company from: {text}"),
])

chain = template | llm | JsonOutputParser()

result = chain.invoke({"text": "Hi, I'm Alex from Acme Corp. alex@acme.com"})
# {'name': 'Alex', 'email': 'alex@acme.com', 'company': 'Acme Corp'}
```

**Pitfall:** The model may produce markdown-wrapped JSON:

````
```json
{"name": "Alex"}
```
````

`JsonOutputParser` strips markdown code fences automatically, but always test with real outputs.

**When to use it:**

- Quick structured extraction in prototypes
- When you do not need field validation
- When the schema is dynamic (not known at code time)

---

## 3. PydanticOutputParser

The most powerful option. Validates the model output against a Pydantic v2 schema
and raises `OutputParserException` with the exact validation failure message if it does not conform.

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, EmailStr, Field

# Step 1: Define what you want back
class ContactCard(BaseModel):
    name: str = Field(description="Full name of the contact")
    email: EmailStr = Field(description="Email address")
    company: str = Field(description="Company or organisation name")
    phone: str | None = Field(None, description="Phone number if present, else null")

# Step 2: Create the parser — it reads the schema to generate format instructions
parser = PydanticOutputParser(pydantic_object=ContactCard)

# Step 3: The parser generates instructions you inject into the prompt
print(parser.get_format_instructions())
# "The output should be formatted as a JSON instance that conforms to the JSON schema..."

# Step 4: Include the format instructions in the prompt
template = ChatPromptTemplate.from_messages([
    ("system", "Extract contact information.\n{format_instructions}"),
    ("human",  "{text}"),
])

chain = template | llm | parser

result = chain.invoke({
    "text": "Hi, I'm Alex from Acme. alex@acme.com, +1-555-9999",
    "format_instructions": parser.get_format_instructions(),
})

print(result)          # ContactCard(name='Alex', email='alex@acme.com', ...)
print(result.email)    # 'alex@acme.com' — typed field, not string slicing
```

### Why Field Descriptions Matter

The model reads your field descriptions to understand intent:

```python
# Vague — model guesses
class Bad(BaseModel):
    d: str   # what is "d"? What format?

# Clear — model extracts correctly
class Good(BaseModel):
    deadline: str = Field(description="ISO 8601 date string, e.g. 2025-03-15. Null if not mentioned.")
```

---

## 4. `.with_structured_output()` — The Modern Preferred Pattern

For all Level 2+ work, prefer `.with_structured_output()` over `PydanticOutputParser`.

```python
from pydantic import BaseModel, Field

class IncidentSummary(BaseModel):
    severity: str = Field(description="P1, P2, P3, or P4")
    affected_service: str = Field(description="Name of the affected service")
    root_cause: str = Field(description="One sentence hypothesis")
    requires_escalation: bool

# The model wrapper handles schema injection internally — no format_instructions needed
structured_llm = llm.with_structured_output(IncidentSummary)

chain = prompt | structured_llm   # no parser needed — structured_llm returns the Pydantic object

result = chain.invoke({"text": alert_text})
print(result.severity)    # "P1"
print(result.requires_escalation)  # True
```

### `PydanticOutputParser` vs `.with_structured_output()`

```
┌───────────────────────────┬────────────────────┬────────────────────────┐
│ Dimension                 │ PydanticOutputParser│ .with_structured_output│
├───────────────────────────┼────────────────────┼────────────────────────┤
│ Schema injection          │ Manual via prompt   │ Handled internally     │
│ Provider support          │ All providers       │ Providers with tool use│
│ Format instructions       │ Explicit in prompt  │ None needed            │
│ Retry on parse failure    │ Manual loop needed  │ Some providers retry   │
│ Best for                  │ Simple prototypes   │ Production pipelines   │
└───────────────────────────┴────────────────────┴────────────────────────┘
```

---

## Retry Logic on Parse Failure

When using `PydanticOutputParser`, the model sometimes produces malformed output.
Use `OutputFixingParser` to automatically retry with the error as feedback:

```python
from langchain.output_parsers import OutputFixingParser

# Wraps PydanticOutputParser — if parsing fails, re-prompts the model with the error
fixing_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=ContactCard),
    llm=llm,
    max_retries=2,
)

chain = template | llm | fixing_parser
```

The retry prompt looks like:

> "The following output failed validation: `{bad_output}`. Error: `{error}`. Return corrected JSON."

---

## Choosing the Right Parser — Decision Tree

```
Does your downstream code need typed fields?
    ├── No  → StrOutputParser
    └── Yes
         │
         Does the schema change at runtime?
         ├── Yes → JsonOutputParser
         └── No
              │
              Are you using a production pipeline (Level 2+)?
              ├── Yes → .with_structured_output(MyModel)
              └── No  → PydanticOutputParser (simpler setup)
```

---

## Common Pitfalls

| Pitfall                                             | What breaks                                | Fix                                                          |
| --------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Not injecting `format_instructions` into the prompt | Parser never sees the schema — random JSON | Always call `parser.get_format_instructions()` and inject it |
| Trusting model output without validation            | Silent data corruption downstream          | Use `PydanticOutputParser` or `.with_structured_output()`    |
| Catching `OutputParserException` silently           | Bad data flows through undetected          | Log the exception + raw output; fail loudly in production    |
| Overly complex nested schemas                       | Model produces incomplete structures       | Start flat; add nesting only when proven necessary           |
| Using `Optional[str]` without a default             | Pydantic raises `ValidationError`          | Set `= Field(None, ...)` as the default                      |

---

## Mini Summary

- `StrOutputParser` is for text responses; fast but unvalidated.
- `JsonOutputParser` gives a dict; useful when the schema is dynamic.
- `PydanticOutputParser` validates against a schema; raises explicit exceptions on failure.
- `.with_structured_output()` is the modern preferred pattern for all production extraction.
- Field descriptions directly improve extraction accuracy — the model reads them.

---

## Next: [05 → Your First Complete Chain](05-first-complete-chain.md)
