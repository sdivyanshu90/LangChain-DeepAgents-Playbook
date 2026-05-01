# 03 — `.with_structured_output()` — How It Works

> **Previous:** [02 → Pydantic v2 Schemas](02-pydantic-v2-schemas.md) | **Next:** [04 → Retry and Fallback](04-retry-and-fallback.md)

---

## Real-World Analogy

When you fill out a government form, you don't write an essay and hope someone extracts
the relevant information. You fill in boxes — each box has a label, a format, and constraints.
The form is designed to be machine-readable.

`.with_structured_output()` gives the model a form to fill in instead of a blank page.
The model fills the boxes; Pydantic checks the entries; your code receives the result.

---

## The Problem with Prompt Engineering Alone

The naive approach to structured output:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract info from the text.
    Return ONLY valid JSON with keys: name, score, tags.
    Example: {{"name": "Alice", "score": 8, "tags": ["python", "ml"]}}
    """),
    ("human", "{text}"),
])
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"text": "..."})
# result is a string — you still need to parse it
```

Problems with this approach:

1. The model may add prose before or after the JSON.
2. The model may use slightly different key names.
3. The model may omit required fields.
4. Your parse step can fail silently.
5. Type coercion does not happen automatically.

---

## How `.with_structured_output()` Actually Works

Under the hood, this method does NOT rely on prompt instructions.
It uses the **tool calling** (function calling) feature of the model API.

```
Step 1: Schema Serialisation
─────────────────────────────
  MySchema ──► JSON Schema ──► Tool definition
  (Pydantic)                   {
                                 "name": "MySchema",
                                 "description": "...",
                                 "parameters": { JSON Schema }
                               }

Step 2: Model API Call
──────────────────────
  The model receives:
  - Your messages
  - A "tools" list containing the schema as a function
  - tool_choice="required" (forced to call this tool)

  The model's job is now to fill in the function arguments — not write prose.

Step 3: Response Parsing
────────────────────────
  Model returns:
  {
    "tool_calls": [{
      "function": {
        "name": "MySchema",
        "arguments": '{"name": "Alice", "score": 8, "tags": ["python"]}'
      }
    }]
  }

Step 4: Validation
──────────────────
  LangChain extracts the arguments JSON.
  Pydantic validates and constructs MySchema(**arguments).
  If validation fails → OutputParserException is raised.
  If it succeeds → you receive a MySchema instance.
```

The model never "decides" the format.
It is forced by the API protocol to produce structured arguments.

---

## Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()   # loads OPENAI_API_KEY from .env

# 1. Define the schema
class ArticleSummary(BaseModel):
    """Summary and metadata extracted from a news article."""
    headline: str = Field(description="A one-sentence headline for the article")
    main_topic: str = Field(description="The primary subject of the article")
    key_points: List[str] = Field(
        description="3-5 most important points from the article",
        min_length=1,
        max_length=5,
    )
    sentiment: str = Field(
        description="Overall tone: 'positive', 'negative', or 'neutral'"
    )

# 2. Create the model and bind the schema
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ArticleSummary)
# ↑ Returns a new Runnable — the original llm is unchanged.
# Every invocation of structured_llm will return an ArticleSummary instance.

# 3. Build the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting structured information from news articles."),
    ("human", "Extract the key information from this article:\n\n{article}"),
])

# 4. Assemble the chain
chain = prompt | structured_llm
# ↑ Note: no output parser needed — structured_llm handles it internally.

# 5. Invoke
result = chain.invoke({"article": "OpenAI announced GPT-5 today..."})

# result is an ArticleSummary instance — NOT a string
print(result.headline)    # "OpenAI Announces GPT-5 with Improved Reasoning"
print(result.key_points)  # ["Faster inference", "Better math", "Multimodal"]
print(type(result))       # <class '__main__.ArticleSummary'>
```

---

## The `mode` Parameter

`.with_structured_output(schema, mode=...)` controls how the output is extracted.

```python
# Default: "tool_calling" — uses function/tool calling API
# Best performance; most providers support this
structured_llm = llm.with_structured_output(
    ArticleSummary,
    method="function_calling",   # explicit (same as default)
)

# "json_mode" — asks the model to return raw JSON, then parses it
# Fallback for models that don't support tool calling
structured_llm = llm.with_structured_output(
    ArticleSummary,
    method="json_mode",
    # When using json_mode, you MUST include the word "JSON" in your prompt,
    # or the model will not enter JSON mode (OpenAI API requirement).
)
```

### When to Use Each Mode

| Mode                           | When to use                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------- |
| `"function_calling"` (default) | GPT-4o, GPT-4o-mini, Claude 3.5+, Gemini Pro — use this always                |
| `"json_mode"`                  | Older model versions; local models via Ollama; any model without tool calling |

---

## Provider Differences

The same `.with_structured_output()` call behaves slightly differently across providers.
LangChain handles this internally, but you should know the differences:

```
┌─────────────────┬──────────────────────────────────────────────┐
│ Provider        │ Behaviour                                    │
├─────────────────┼──────────────────────────────────────────────┤
│ OpenAI (GPT-4o) │ Uses "tools" API with tool_choice="required" │
│                 │ Most reliable; strict mode available          │
├─────────────────┼──────────────────────────────────────────────┤
│ Anthropic       │ Uses "tools" API                             │
│ (Claude 3.5+)   │ Reliable; max_tokens must be set explicitly  │
├─────────────────┼──────────────────────────────────────────────┤
│ Google Gemini   │ Uses function calling                        │
│                 │ Minor differences in enum handling            │
├─────────────────┼──────────────────────────────────────────────┤
│ Ollama (local)  │ Uses json_mode; format instructions in prompt│
│                 │ Lower reliability for complex schemas         │
└─────────────────┴──────────────────────────────────────────────┘
```

---

## `include_raw` — Getting Both Validated and Raw Output

Sometimes you need the raw model response alongside the parsed object
(for debugging, logging, or fallback):

```python
structured_llm = llm.with_structured_output(
    ArticleSummary,
    include_raw=True,   # return both parsed and raw output
)

result = structured_llm.invoke([HumanMessage(content="...")])
# result is now a dict:
# {
#   "raw":    AIMessage(content="", tool_calls=[...]),  ← raw API response
#   "parsed": ArticleSummary(headline="...", ...),      ← validated object
#   "parsing_error": None                               ← or an exception
# }

# Access the validated object
article = result["parsed"]

# Access the raw response (for debugging)
raw = result["raw"]
print(raw.tool_calls)
```

This is the recommended pattern for production: always log `raw` for observability.

---

## Passing a Dict Schema Instead of a Pydantic Model

For simple cases or dynamic schemas, you can pass a JSON Schema dict directly:

```python
schema_dict = {
    "title": "QuickExtract",
    "type": "object",
    "properties": {
        "name":  {"type": "string", "description": "Person's full name"},
        "age":   {"type": "integer", "description": "Age in years"},
    },
    "required": ["name", "age"],
}

structured_llm = llm.with_structured_output(schema_dict)
result = structured_llm.invoke([HumanMessage(content="Alice is 30 years old.")])
# result is a dict: {"name": "Alice", "age": 30}
# No Pydantic validation — you get a plain dict
```

Use Pydantic models in production (stronger validation).
Use dict schemas for prototyping or when the schema is generated dynamically.

---

## Using with the Model Factory Pattern

Following the repository convention for provider-agnostic code:

```python
# shared/utils/model_factory.py provides get_chat_model()
from shared.utils.model_factory import get_chat_model
from pydantic import BaseModel, Field

class ExtractedFact(BaseModel):
    subject: str = Field(description="What the fact is about")
    claim: str = Field(description="The specific claim being made")
    source_sentence: str = Field(description="The exact sentence from the source")

# Works regardless of MODEL_PROVIDER in .env
llm = get_chat_model(temperature=0)
structured_llm = llm.with_structured_output(ExtractedFact)
```

---

## Common Pitfalls

| Pitfall                                             | What goes wrong                                                             | Fix                                                                   |
| --------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Using `with_structured_output` on a streaming chain | Streaming and structured output conflict — you get an error or partial JSON | Disable streaming when using structured output                        |
| Not setting `temperature=0`                         | Stochastic outputs may produce different field values on retries            | Always use `temperature=0` for extraction and classification          |
| Expecting a string response                         | `.with_structured_output()` returns a Pydantic object, not a string         | Access `.field_name` on the result; don't call `.content`             |
| Using `json_mode` without "JSON" in the prompt      | OpenAI's JSON mode is not activated — returns normal text                   | Add "Return your answer as JSON" to the system prompt                 |
| Deep nesting in a dict schema                       | No validation on nested types; silent coercion failures                     | Use Pydantic models for production; dict schemas for prototyping only |

---

## Mini Summary

- `.with_structured_output()` uses tool calling internally — not prompt engineering.
- The model fills in structured "function arguments", not free-form prose.
- Pydantic validates the arguments before your code receives them.
- Use `method="function_calling"` (default) for production; `"json_mode"` for local models.
- Set `include_raw=True` in production to capture the raw response for debugging.
- The result is a Pydantic model instance — access fields with dot notation, not `.content`.
