[← Async Tools](02-async-tools.md) | [Next → Testing Tools in Isolation](04-tool-testing-in-isolation.md)

---

# 03 — Tool Exception Patterns

## The Problem with Unhandled Exceptions

When a tool raises an unhandled exception in an agent, one of two things happens:

1. The exception propagates up and **crashes the entire graph run** — every other step is lost
2. The framework catches it silently and the model receives **no ToolMessage** — violating the message contract and causing an API error on the next call

Neither outcome is acceptable in production. The goal of this file is to give you a
complete vocabulary for _expected_ and _unexpected_ tool failures and patterns for
handling each one.

---

## Real-World Analogy

An air traffic controller's console can show three states for a flight:

- **Green (nominal):** Flight is on schedule
- **Yellow (advisory):** Flight is delayed — here's the reason, here are options
- **Red (emergency):** Flight has a critical issue — immediate escalation required

Your tools should work the same way. Every result should signal one of three states:
success, expected failure (with recovery hint), or unexpected failure (needs investigation).
_Silent_ failure is the fourth state you must eliminate.

---

## The Three Exception Strategies

### Strategy 1: `ToolException` for Expected Failures

Use `ToolException` when the failure is predictable and the model can act on it:

```python
from langchain_core.tools import tool, ToolException
import httpx

@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """
    Get the current exchange rate between two currencies.
    Returns a JSON string with base, target, and rate.
    from_currency and to_currency should be 3-letter ISO 4217 codes (e.g. 'USD', 'EUR', 'GBP').
    """
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    # Expected validation failure:
    if len(from_currency) != 3 or len(to_currency) != 3:
        raise ToolException(
            f"Currency codes must be 3 letters. Got '{from_currency}' and '{to_currency}'. "
            "Examples: 'USD', 'EUR', 'GBP', 'JPY'."
        )

    try:
        resp = httpx.get(
            f"https://api.exchangerate-api.com/v4/latest/{from_currency}",
            timeout=8.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Expected: unsupported currency code
            raise ToolException(
                f"Currency '{from_currency}' is not supported. "
                "Use standard ISO 4217 codes like 'USD', 'EUR', 'GBP'."
            )
        # Unexpected: server error — fall through to generic handler
        raise

    data = resp.json()
    rate = data["rates"].get(to_currency)
    if rate is None:
        raise ToolException(
            f"Target currency '{to_currency}' not found. Verify the currency code."
        )

    import json
    return json.dumps({"base": from_currency, "target": to_currency, "rate": rate})
```

### Strategy 2: Custom `handle_tool_error` Callback

When you need to format the error message for the model in a specific way:

```python
from langchain_core.tools import ToolException

def exchange_rate_error_handler(error: Exception) -> str:
    """Format errors from get_exchange_rate into model-friendly messages."""
    if isinstance(error, ToolException):
        return (
            f"Exchange rate lookup failed: {error}\n"
            "Action: Correct the currency codes and retry. "
            "If the error persists, inform the user that exchange rate data is unavailable."
        )
    # Unexpected errors
    return (
        f"An unexpected error occurred in the exchange rate tool: {type(error).__name__}. "
        "Action: Retry once. If it fails again, inform the user."
    )

@tool(handle_tool_error=exchange_rate_error_handler)
def get_exchange_rate_safe(from_currency: str, to_currency: str) -> str:
    """
    Get the current exchange rate. from_currency and to_currency are 3-letter ISO 4217 codes.
    Returns JSON with base, target, and rate.
    """
    # ... same implementation as above ...
    pass
```

### Strategy 3: The Safe Tool Pattern — Never Raises

Use this pattern when you need full control over the output structure,
or when using the tool in a framework that doesn't support `handle_tool_error`:

```python
import json
import traceback
from typing import Optional
from langchain_core.tools import tool

@tool
def safe_fetch_user(user_id: str) -> str:
    """
    Safely retrieve a user record by ID.
    Always returns a JSON string with 'success', 'data', and 'error' fields.
    success=true means data is populated; success=false means error explains the failure.
    user_id should be a UUID string.
    """
    import uuid
    # Input validation
    try:
        uuid.UUID(user_id)
    except ValueError:
        return json.dumps({
            "success": False,
            "data": None,
            "error": f"Invalid user_id '{user_id}'. Must be a UUID, e.g. '550e8400-e29b-41d4-a716-446655440000'.",
            "recoverable": True,
        })

    try:
        # Simulate DB call
        user_data = {"id": user_id, "name": "Alice", "email": "alice@example.com"}
        return json.dumps({"success": True, "data": user_data, "error": None})
    except Exception as e:
        # Log the full traceback for observability (doesn't affect the model's message)
        print(f"[ERROR] safe_fetch_user: {traceback.format_exc()}")
        return json.dumps({
            "success": False,
            "data": None,
            "error": f"Database error: {str(e)}",
            "recoverable": False,
        })
```

---

## Returning Partial Results vs Hard Failures

For batch operations, returning partial results is almost always more useful than failing entirely:

```python
import json
from langchain_core.tools import tool

@tool
def batch_geocode(addresses: list[str]) -> str:
    """
    Convert a list of street addresses to latitude/longitude coordinates.
    Returns a JSON object with 'results' (list of {address, lat, lng}) and 'failed' (list of {address, error}).
    Partial results are returned even if some addresses fail.
    """
    results = []
    failed = []

    for addr in addresses:
        try:
            # Stub: replace with real geocoding API call
            if not addr.strip():
                raise ValueError("Empty address")
            lat, lng = 48.8566 + len(addr) * 0.0001, 2.3522   # mock
            results.append({"address": addr, "lat": lat, "lng": lng})
        except Exception as e:
            failed.append({"address": addr, "error": str(e)})

    return json.dumps({
        "results": results,
        "failed": failed,
        "partial": len(failed) > 0,
        "total": len(addresses),
        "succeeded": len(results),
    })
```

---

## Decision Tree: Which Pattern to Use?

```
Is the failure predictable from input validation or known API constraints?
    YES → raise ToolException("clear explanation + how to fix")
    NO  → continue

Is the tool used in a context where handle_tool_error is available?
    YES → @tool(handle_tool_error=my_formatter)
    NO  → continue

Is the tool used in a custom LangGraph node or non-standard framework?
    YES → Use the safe-tool pattern (never raises, always returns structured JSON)
    NO  → @tool(handle_tool_error=True) is a reasonable default

Is it a batch operation where some items may succeed?
    → Always use partial results pattern (return both results and failed lists)
```

---

## Testing Error Paths (Preview of Module 3.4)

```python
import json
import pytest
from langchain_core.tools import ToolException

def test_exchange_rate_invalid_code():
    """Tool should raise ToolException for non-3-letter currency code."""
    with pytest.raises(ToolException) as exc_info:
        get_exchange_rate.invoke({"from_currency": "US", "to_currency": "EUR"})
    assert "3 letters" in str(exc_info.value)

def test_safe_fetch_user_invalid_uuid():
    """Safe tool should return success=False for invalid UUID without raising."""
    result = json.loads(safe_fetch_user.invoke({"user_id": "not-a-uuid"}))
    assert result["success"] is False
    assert "UUID" in result["error"]

def test_batch_geocode_partial_failure():
    """Batch tool should return partial results when some addresses fail."""
    result = json.loads(batch_geocode.invoke({"addresses": ["Paris, France", ""]}))
    assert result["partial"] is True
    assert result["succeeded"] == 1
    assert len(result["failed"]) == 1
```

---

## Common Pitfalls

| Pitfall                                                    | Symptom                                        | Fix                                                                                |
| ---------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| Raising `ToolException` for truly unexpected errors        | Hides bugs; model retries forever              | Only raise `ToolException` for recoverable, expected failures                      |
| Silent `except: pass` in tool body                         | Model receives empty ToolMessage; hallucinates | Always return a structured error message                                           |
| `handle_tool_error=True` without testing the error message | Model receives unhelpful Python traceback      | Always test what the model sees; use a custom formatter                            |
| Partial results with no `"partial"` signal                 | Model doesn't know results are incomplete      | Include `"partial": true/false` in every batch tool response                       |
| Raising in `__init__` or class constructor of a tool       | Error at bind time, not call time              | Validate configuration at bind time but raise `ToolException` for runtime failures |

---

## Mini Summary

- `ToolException` signals an expected, model-actionable failure; it should include a clear recovery hint
- `handle_tool_error=True` (or a custom callback) converts exceptions to `ToolMessage` content automatically
- The safe-tool pattern — never raises, always returns structured JSON — is the most robust for complex frameworks
- Batch tools should return `{"results": [...], "failed": [...], "partial": bool}` instead of failing entirely
- Test your error paths in isolation before wiring the tool to an agent

---

[← Async Tools](02-async-tools.md) | [Next → Testing Tools in Isolation](04-tool-testing-in-isolation.md)
