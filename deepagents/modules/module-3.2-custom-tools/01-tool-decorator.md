[← Module Overview](README.md) | [Next → Async Tools](02-async-tools.md)

---

# 01 — The `@tool` Decorator in Depth

## Why the Decorator Exists

Without `@tool`, you would need to manually subclass `BaseTool`, implement `_run()` and
`_arun()`, write a Pydantic schema class, and wire them together. The `@tool` decorator
collapses that boilerplate into a single annotation while still giving you full control
over the schema when you need it.

---

## Real-World Analogy

Consider a REST API gateway. You write a Python function (the handler), annotate it with
`@app.route("/weather")`, and the framework handles URL parsing, request deserialization,
response serialization, and error routing.

`@tool` is the same pattern: you write the business logic as a plain function, annotate it,
and LangChain handles schema generation, argument validation, and invocation orchestration.

---

## The Four Forms of `@tool`

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

# ── Form 1: Simplest — inferred from type hints ─────────────────────────────
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together. Returns the sum as a float."""
    return a + b

# ── Form 2: Custom name ──────────────────────────────────────────────────────
@tool("sum_values")
def add_numbers_v2(a: float, b: float) -> float:
    """Add two values and return their sum."""
    return a + b

# ── Form 3: Explicit Pydantic schema ─────────────────────────────────────────
class SearchInput(BaseModel):
    query: str = Field(description="The search query. Be specific; avoid single-word queries.")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results to return (1-20).")
    language: str = Field(default="en", description="ISO 639-1 language code, e.g. 'en', 'fr', 'de'.")

@tool("search_knowledge_base", args_schema=SearchInput)
def search_knowledge_base(query: str, max_results: int = 5, language: str = "en") -> str:
    """
    Search the internal knowledge base.
    Use this when the user asks about company policies, product docs, or internal procedures.
    Returns a list of relevant document snippets.
    """
    return f"[STUB] {max_results} results for '{query}' in language '{language}'"

# ── Form 4: With handle_tool_error ───────────────────────────────────────────
from langchain_core.tools import ToolException

@tool(handle_tool_error=True)
def get_record(record_id: str) -> str:
    """Retrieve a record from the data warehouse by its unique ID (e.g. 'REC-12345')."""
    if not record_id.startswith("REC-"):
        raise ToolException(
            f"Invalid record_id '{record_id}'. ID must start with 'REC-', e.g. 'REC-12345'."
        )
    return f"Record {record_id}: {{ 'status': 'active', 'owner': 'alice@example.com' }}"
```

---

## Why Pydantic Beats Raw Type Hints for Tools

Raw type hints only express _type_. Pydantic `Field` expresses:

- Human-readable `description` (model reads this to fill the argument correctly)
- Validation constraints (`ge`, `le`, `min_length`, `max_length`, `pattern`)
- Default values with semantic meaning
- Example values that appear in the schema

```python
from pydantic import BaseModel, Field

# Type hint only — minimal schema sent to model:
# {"city": {"title": "City", "type": "string"}}

# Pydantic Field — rich schema:
class WeatherInput(BaseModel):
    city: str = Field(
        description="The city name in English, e.g. 'Paris', 'New York', 'Tokyo'.",
        min_length=2,
        max_length=100,
        examples=["Paris", "London", "Sydney"],
    )
    units: str = Field(
        default="celsius",
        description="Temperature unit: 'celsius' or 'fahrenheit'.",
        pattern="^(celsius|fahrenheit)$",
    )

# The resulting JSON schema includes description, examples, and pattern — the model
# uses ALL of this to fill arguments correctly.
```

### Validating Arguments Before the Function Runs

LangChain calls `tool.invoke(args)`, which runs Pydantic validation on `args` before
passing them to your function. Invalid inputs raise `ValidationError` before your code runs:

```python
import pytest
from pydantic import ValidationError

# This will raise a Pydantic ValidationError before get_weather's body executes:
try:
    get_weather.invoke({"city": "", "units": "kelvin"})
except Exception as e:
    print(type(e))   # pydantic.ValidationError — units doesn't match the pattern
```

---

## Docstring Quality — Bad vs Good

The docstring is injected verbatim into the `description` field of the tool schema.
The model reads it to decide _whether_ to call the tool and _how_ to fill arguments.

```python
# ❌ BAD — gives the model almost no information
@tool
def fetch_order(order_id: str) -> str:
    """Fetch an order."""
    ...

# ❌ BAD — long but still wrong: describes implementation, not usage
@tool
def fetch_order(order_id: str) -> str:
    """
    This function connects to the orders database using SQLAlchemy, runs a SELECT
    query on the orders table, and returns the result as a JSON string.
    """
    ...

# ✅ GOOD — answers: what does it return, when should I use it, what format is the ID
@tool
def fetch_order(order_id: str) -> str:
    """
    Retrieve a customer order by its ID.
    Returns a JSON string with keys: order_id, status, items, total_usd, created_at.
    Use this when the user asks about a specific order status or details.
    order_id format: 'ORD-' followed by 8 digits, e.g. 'ORD-00012345'.
    """
    ...
```

**Rule of thumb for docstrings:**

1. First sentence: what it returns
2. Second sentence: when to use it (and when NOT to)
3. Third sentence: argument format / constraints

---

## Return Type Annotations

The return type annotation affects how LangChain handles the output:

```python
from typing import Optional

# str return — content goes directly into ToolMessage as-is
@tool
def simple_tool(x: str) -> str:
    return "result"

# dict return — LangChain serialises to JSON string for ToolMessage
@tool
def dict_tool(x: str) -> dict:
    return {"key": "value", "count": 42}

# Optional — tool may return None; treat as empty result
@tool
def nullable_tool(x: str) -> Optional[str]:
    return None if x == "skip" else "result"
```

Always prefer returning a `str` (or a JSON-stringified dict via `json.dumps()`). This
makes ToolMessage content fully transparent in traces and logs.

---

## Inspecting a Tool After Creation

```python
@tool
def example_tool(name: str, limit: int = 10) -> str:
    """Fetch up to `limit` items for the given `name`. Returns a JSON array."""
    return "[]"

# These attributes are always available on a @tool-decorated function:
print(example_tool.name)          # "example_tool"
print(example_tool.description)   # "Fetch up to `limit` items for the given `name`..."
print(example_tool.args)          # {'name': {'title': 'Name', 'type': 'string'}, ...}

# Full JSON schema (what gets sent to the API):
import json
print(json.dumps(example_tool.args_schema.model_json_schema(), indent=2))

# Direct invocation (used in tests):
result = example_tool.invoke({"name": "widgets", "limit": 5})
print(result)   # "[]"

# Async invocation:
import asyncio
result = asyncio.run(example_tool.ainvoke({"name": "widgets"}))
```

---

## Building a Multi-Field Tool — Complete Example

```python
import json
import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool, ToolException

class GitHubSearchInput(BaseModel):
    query: str = Field(
        description="GitHub search query, e.g. 'langchain agent stars:>100'.",
        min_length=3,
    )
    repo_type: str = Field(
        default="repositories",
        description="What to search: 'repositories', 'issues', or 'code'.",
        pattern="^(repositories|issues|code)$",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Maximum number of results to return.",
    )

@tool("search_github", args_schema=GitHubSearchInput, handle_tool_error=True)
def search_github(query: str, repo_type: str = "repositories", max_results: int = 5) -> str:
    """
    Search GitHub for repositories, issues, or code snippets.
    Returns a JSON array of results with name, url, description, and star_count.
    Use this when the user wants to find open-source projects, code examples, or issues.
    GitHub token is read from the GITHUB_TOKEN environment variable.
    """
    import os
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        resp = httpx.get(
            f"https://api.github.com/search/{repo_type}",
            params={"q": query, "per_page": max_results},
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        raise ToolException("GitHub API timed out. Please retry.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            raise ToolException("GitHub rate limit exceeded. Wait 60 seconds before retrying.")
        raise ToolException(f"GitHub API error: HTTP {e.response.status_code}")

    data = resp.json()
    items = data.get("items", [])[:max_results]
    results = [
        {
            "name": item.get("full_name") or item.get("title", ""),
            "url": item.get("html_url", ""),
            "description": item.get("description") or item.get("body", "")[:200],
            "stars": item.get("stargazers_count"),
        }
        for item in items
    ]
    return json.dumps(results, ensure_ascii=False)
```

---

## Common Pitfalls

| Pitfall                                            | Symptom                                                  | Fix                                                                      |
| -------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| Docstring describes implementation, not interface  | Model calls wrong tool or fills args incorrectly         | Focus docstring on: what it returns, when to use it, arg format          |
| Using `str` type hint without `Field.description`  | Model guesses argument format (often wrong)              | Add `Field(description=...)` for every non-obvious argument              |
| Returning a Python dict without `json.dumps`       | ToolMessage content is `"{'key': 'val'}"` (invalid JSON) | Always `return json.dumps(result)` for dict outputs                      |
| `@tool` on a method instead of a function          | `self` becomes an argument in the schema                 | Extract to a module-level function or use `StructuredTool.from_function` |
| Missing `handle_tool_error=True` on a fragile tool | Exception propagates, crashes the agent                  | Add `handle_tool_error=True` or use the safe-tool pattern                |

---

## Mini Summary

- `@tool` collapses `BaseTool` boilerplate into a decorator; use `args_schema=PydanticModel` for rich schemas
- Pydantic `Field(description=...)` is not optional — the model reads it to fill arguments correctly
- Docstrings should answer: what does it return, when to use it, what format are the arguments
- Return `str` or `json.dumps(dict)` — avoid returning raw Python dicts
- Inspect `.name`, `.description`, `.args`, `.args_schema.model_json_schema()` to verify what the model sees

---

[← Module Overview](README.md) | [Next → Async Tools](02-async-tools.md)
