# 01 — Why Structured Output

> **Previous:** [README → Module Index](README.md) | **Next:** [02 → Pydantic v2 Schemas](02-pydantic-v2-schemas.md)

---

## Real-World Analogy

Imagine you hire a contractor and ask them to give you a quote.
If they hand back a paragraph of prose — "The job will probably run around eight to ten thousand,
depending on materials, maybe more if we hit problems with the plumbing" —
you cannot enter that into your accounting system.

You needed a form: `{"labor": 7000, "materials": 2500, "contingency": 500}`.

An LLM is exactly like that contractor.
The _information_ it produces might be correct.
But unless the **shape** is guaranteed, your application cannot use it.

---

## The Core Problem: Free-Form Text Breaks Software

When an LLM returns a plain string, every caller must parse it differently.
Parsing logic is fragile and multiplies across the codebase.

```
┌──────────────────────────────────────────────────────────┐
│  LLM output (free-form string)                           │
│                                                          │
│  "Based on the review, the sentiment is positive,        │
│   the score would be around 8/10, and the topics        │
│   covered are price, delivery, and customer service."    │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────┐
  │  Your parsing code     │
  │                        │
  │  re.search(r"(\d+)/10")│  ← FRAGILE: "around 8 out of 10" breaks this
  │  if "positive" in ...  │  ← FRAGILE: "not entirely positive" is a bug
  │  split("and")[1]       │  ← FRAGILE: any list variation breaks it
  └────────────────────────┘
```

This is the silent-failure problem.
Your regex finds `8` and `10` — it does not fail, it does not throw.
It just gives you a wrong answer.

---

## Why "It Works in Testing" Is Dangerous

Models are probabilistic.
In development you might test 20 prompts and they all return a parseable format.
In production, the 500th prompt hits a slightly unusual phrasing and the format breaks.

```
   Development environment:
   ┌──────────────┐    ┌──────────────┐
   │ Test input 1 │ ─► │ "score: 8"   │  ← your regex finds it ✓
   │ Test input 2 │ ─► │ "Score: 7"   │  ← your regex finds it ✓
   └──────────────┘    └──────────────┘

   Production (10,000 requests):
   ┌──────────────┐    ┌──────────────────────────────────┐
   │ Input #412   │ ─► │ "I'd give it roughly eight out   │  ← regex returns None
   │              │    │  of ten stars for the quality."  │  ← code does result["score"]
   └──────────────┘    └──────────────────────────────────┘
                                                            ↓
                                                     KeyError or None
                                                     silently propagates
```

The failure is **silent** and **deferred** — it may surface as a downstream bug
in a database write, a corrupt report, or a wrong recommendation.

---

## What "Contract" Means Here

A schema is a contract with three parties:

| Party         | Role                                                            |
| ------------- | --------------------------------------------------------------- |
| **You**       | Define the expected structure and types                         |
| **The model** | Is instructed to produce output matching that structure         |
| **Pydantic**  | Validates the output before it ever reaches your business logic |

If the model violates the contract, Pydantic raises an exception **immediately**,
at the boundary, before the bad data can travel further.
This is the **fail-fast** principle applied to LLM output.

```
Without schema:                    With schema:
                                                       ┌──────────────┐
LLM → string → your code          LLM → JSON → Pydantic │ raises here  │
        ↓                                               └──────────────┘
  bad data silently               Bad data never reaches your logic.
  corrupts state                  Error is explicit, catchable, local.
```

---

## The Four Guarantees Structured Output Provides

### 1. Field Existence

Every field you declare **will exist** on the returned object.
You never write `result.get("score", 0)` defensively — the field is always there.

### 2. Type Safety

A field declared as `int` will be an integer.
The string `"8"` will be coerced to `8`.
A field declared as `list[str]` will be a list, never a bare string.

### 3. Validation Rules

You can add constraints: minimum value, maximum length, regex pattern, enum membership.
Pydantic enforces them — the LLM cannot bypass them.

### 4. Documentation as Code

Your schema is self-documenting.
Any developer reading `class SentimentResult(BaseModel)` immediately knows
exactly what the model is supposed to return.

---

## Cost of NOT Using Structured Output

Let's quantify the real cost:

```
Scenario: Review analysis pipeline processing 50,000 reviews/day

Without schema:
  - 0.2% parse failure rate (seems low)
  - 0.002 × 50,000 = 100 corrupted records/day
  - Each corrupt record silently written to database
  - Discovered 3 weeks later during reporting
  - Manual remediation cost: 2 engineer-days
  - Reputation cost: dashboard showed wrong sentiment for a major product

With schema:
  - Parse failure → exception → retry → log → alert
  - 0 corrupt records in database
  - Failures visible in real time
  - Remediation: fix the schema or prompt in 2 hours
```

Silent failures compound over time.
Visible failures get fixed immediately.

---

## The Three Approaches (Overview)

There are three ways to get structured output from a LangChain model.
They differ in how much control they give you and how much they rely on the model following instructions.

```
Approach 1: Prompt Engineering
──────────────────────────────
  System: "Return JSON with keys: name, score, tags"
  Risk: model may add prose, skip fields, use wrong types
  When to use: never in production for machine-consumed output

Approach 2: Output Parsers
──────────────────────────
  model | PydanticOutputParser(pydantic_object=MySchema)
  Risk: depends on model following format instructions in prompt
  When to use: models without native tool calling (rare)

Approach 3: .with_structured_output()    ← the right way
─────────────────────────────────────────────────────────
  llm.with_structured_output(MySchema)
  Uses tool calling: model fills in a structured form, not prose
  Validation: Pydantic validates before you touch the result
  When to use: always, for any production extraction/classification
```

Approach 3 is what this module teaches.
The others are covered only so you recognise them in legacy code.

---

## Common Pitfalls

| Pitfall                                        | What goes wrong                                               | How to avoid it                                            |
| ---------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- |
| Parsing the model's string output with regex   | Silent failures when the model varies phrasing                | Use `.with_structured_output()` instead                    |
| Using `Optional` on every field "just in case" | You never know if the field was absent or the model forgot it | Only mark truly optional fields as `Optional`              |
| Trusting that testing covers all formats       | Models are probabilistic; test set is not representative      | Use schema + validation to enforce the contract at runtime |
| Putting too many fields in one schema          | Model gets confused; some fields are hallucinated             | Split into focused schemas with 3-7 fields each            |
| Not reading the validation error message       | Retrying the same prompt that will always fail                | Log and inspect `ValidationError` details before retrying  |

---

## Mini Summary

- Free-form LLM output is useful for humans but brittle for software.
- Silent failures are worse than loud failures: they corrupt data silently.
- A schema is a contract: you define the shape, the model fills it, Pydantic validates it.
- `.with_structured_output()` uses tool calling internally — it is not prompt engineering.
- Structured output is not optional in production; it is the minimum baseline for reliability.

The next topic covers how to write good Pydantic v2 schemas that make the contract as strong as possible.
