[← Tool Exception Patterns](03-tool-exception-patterns.md) | [Next → Tool Registry Pattern](05-tool-registry-pattern.md)

---

# 04 — Testing Tools in Isolation

## Why Test Tools Before the Agent?

An agent test is a black-box integration test. When it fails, you don't know if the failure
is in the model's reasoning, the tool's logic, the error handling, or the message wiring.

Testing tools in isolation collapses that uncertainty: you know exactly what a tool does
given specific inputs before you give the model any influence over it. Bugs found in isolation
are fixed in minutes. Bugs found through agent tests take hours to diagnose.

**Principle:** Test the tool. Wire it to the agent. Trust the wiring.

---

## Real-World Analogy

A Formula 1 team doesn't test the entire car on the track before verifying every component
on the bench. The brake callipers are tested under load in isolation. The fuel injectors
are calibrated before installation. Only when every component is verified does the car
go to the track.

Testing a tool before wiring it to an agent is the component bench test.

---

## Project Setup

```
tests/
  tools/
    test_weather_tool.py
    test_search_tool.py
    test_database_tool.py
  conftest.py       # shared fixtures
pytest.ini           # asyncio mode configuration
```

`pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
```

`tests/conftest.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def mock_http_session():
    """Fixture that provides a pre-configured mock aiohttp session."""
    with patch("aiohttp.ClientSession") as mock_cls:
        mock_session = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_session
        yield mock_session
```

---

## Pattern 1 — Testing Basic Input/Output

```python
# tests/tools/test_weather_tool.py
import json
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.tools import ToolException

# Import the tool under test — note: testing the tool directly, not through an agent
from myapp.tools.weather import get_weather

class TestGetWeatherTool:
    """Tests for the get_weather tool in isolation."""

    def test_returns_valid_json(self):
        """Tool should return a parseable JSON string on success."""
        with patch("myapp.tools.weather.httpx.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "current": {"temp_c": 18.0, "condition": {"text": "Partly cloudy"}}
            }
            mock_get.return_value.raise_for_status = MagicMock()

            result = get_weather.invoke({"city": "Paris"})
            parsed = json.loads(result)

            assert "temperature_celsius" in parsed
            assert "condition" in parsed
            assert parsed["temperature_celsius"] == 18.0

    def test_city_name_is_preserved(self):
        """Tool should include the queried city in the response."""
        with patch("myapp.tools.weather.httpx.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "current": {"temp_c": 25.0, "condition": {"text": "Sunny"}}
            }
            mock_get.return_value.raise_for_status = MagicMock()

            result = get_weather.invoke({"city": "Tokyo"})
            parsed = json.loads(result)
            assert parsed.get("city") == "Tokyo"
```

---

## Pattern 2 — Testing Input Validation

```python
    def test_empty_city_raises_tool_exception(self):
        """Tool should raise ToolException for empty city string."""
        with pytest.raises(ToolException) as exc_info:
            get_weather.invoke({"city": ""})
        # Verify the error message is model-actionable:
        assert "city" in str(exc_info.value).lower()
        assert "empty" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()

    def test_very_long_city_name_raises(self):
        """Tool should reject city names over 100 characters."""
        long_city = "A" * 101
        with pytest.raises(Exception):   # Pydantic ValidationError from schema
            get_weather.invoke({"city": long_city})

    def test_numeric_city_is_rejected(self):
        """Tool should not silently accept numeric-only city names."""
        with pytest.raises((ToolException, Exception)):
            get_weather.invoke({"city": "12345"})
```

---

## Pattern 3 — Mocking External APIs

```python
# tests/tools/test_search_tool.py
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp

from myapp.tools.search import search_web_async

class TestSearchWebAsyncTool:
    """Tests for the async web search tool."""

    async def test_returns_list_of_results(self):
        """Async tool should return a JSON array of search results."""
        mock_response_data = {
            "web": {
                "results": [
                    {"title": "LangChain Docs", "url": "https://docs.langchain.com", "description": "Official docs"},
                    {"title": "LangGraph Guide", "url": "https://langchain-ai.github.io/langgraph", "description": "Graph guide"},
                ]
            }
        }
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_response_data
            mock_resp.raise_for_status = MagicMock()
            mock_session.get.return_value.__aenter__.return_value = mock_resp

            result = await search_web_async.ainvoke({"query": "LangChain agents", "num_results": 2})
            parsed = json.loads(result)

            assert isinstance(parsed, list)
            assert len(parsed) == 2
            assert parsed[0]["title"] == "LangChain Docs"
            assert "url" in parsed[0]
            assert "snippet" in parsed[0]

    async def test_handles_empty_results(self):
        """Tool should return empty list when no search results found."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_resp = AsyncMock()
            mock_resp.json.return_value = {"web": {"results": []}}
            mock_resp.raise_for_status = MagicMock()
            mock_session.get.return_value.__aenter__.return_value = mock_resp

            result = await search_web_async.ainvoke({"query": "xkcd12345nonexistent"})
            parsed = json.loads(result)
            assert parsed == []

    async def test_timeout_returns_error_not_crash(self):
        """Tool should handle network timeouts gracefully without raising."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session.get.side_effect = aiohttp.ServerTimeoutError()

            # The tool uses the safe-tool pattern — should not raise
            result = await search_web_async.ainvoke({"query": "test query"})
            parsed = json.loads(result)
            assert parsed.get("error") is not None
```

---

## Pattern 4 — Testing Error Paths

```python
# tests/tools/test_database_tool.py
import json
import pytest
from unittest.mock import patch
from langchain_core.tools import ToolException

from myapp.tools.database import safe_fetch_user, fetch_order

class TestDatabaseTools:

    def test_safe_fetch_user_invalid_uuid(self):
        """Safe tool returns structured error for invalid UUID, never raises."""
        result = json.loads(safe_fetch_user.invoke({"user_id": "not-a-uuid"}))
        assert result["success"] is False
        assert result["error"] is not None
        assert result["recoverable"] is True   # model can retry with correct ID

    def test_safe_fetch_user_db_error(self):
        """Safe tool returns success=False when DB raises, does not propagate exception."""
        with patch("myapp.tools.database.db_session.query") as mock_query:
            mock_query.side_effect = RuntimeError("DB connection lost")
            result = json.loads(safe_fetch_user.invoke({"user_id": "550e8400-e29b-41d4-a716-446655440000"}))
            assert result["success"] is False
            assert "error" in result

    def test_fetch_order_raises_tool_exception_for_invalid_format(self):
        """fetch_order should raise ToolException (not ValueError) for bad ID format."""
        with pytest.raises(ToolException) as exc_info:
            fetch_order.invoke({"order_id": "12345"})   # missing 'ORD-' prefix
        # Error message must be model-actionable:
        error_msg = str(exc_info.value)
        assert "ORD-" in error_msg or "format" in error_msg.lower()

    def test_tool_call_id_not_required_for_direct_invoke(self):
        """Verify that direct .invoke() works without agent context."""
        # Tools should be testable without a ToolCall wrapper
        result = safe_fetch_user.invoke({"user_id": "550e8400-e29b-41d4-a716-446655440000"})
        assert isinstance(result, str)
        json.loads(result)   # should be valid JSON
```

---

## Pattern 5 — Testing the Schema Itself

```python
# tests/tools/test_tool_schemas.py
import json
from myapp.tools.weather import get_weather
from myapp.tools.search import search_web_async

class TestToolSchemas:

    def test_weather_tool_has_required_name(self):
        assert get_weather.name == "get_weather"

    def test_weather_tool_description_mentions_return_format(self):
        """Model needs to know the return format from the description."""
        desc = get_weather.description
        assert "json" in desc.lower() or "JSON" in desc

    def test_weather_tool_schema_has_city_with_description(self):
        schema = get_weather.args_schema.model_json_schema()
        city_prop = schema["properties"]["city"]
        assert "description" in city_prop, "city field must have a Pydantic Field description"
        assert len(city_prop["description"]) > 10, "description is too short to be useful"

    def test_all_required_fields_are_documented(self):
        """Every required field should have a non-empty Field description."""
        schema = get_weather.args_schema.model_json_schema()
        required = schema.get("required", [])
        props = schema.get("properties", {})
        for field_name in required:
            prop = props.get(field_name, {})
            desc = prop.get("description", "")
            assert len(desc) > 5, f"Required field '{field_name}' has no useful description"
```

---

## Running the Test Suite

```bash
# Run all tool tests:
pytest tests/tools/ -v

# Run only async tests:
pytest tests/tools/ -v -k "async"

# Run with coverage:
pytest tests/tools/ --cov=myapp.tools --cov-report=term-missing

# Run a single test class:
pytest tests/tools/test_weather_tool.py::TestGetWeatherTool -v
```

---

## Common Pitfalls

| Pitfall                                         | Symptom                                                     | Fix                                                               |
| ----------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| Testing the tool through an agent               | Can't isolate which component failed                        | Call `tool.invoke(args)` directly; no LLM needed                  |
| Not mocking external APIs                       | Tests hit real endpoints, flaky and slow                    | Always mock network calls with `unittest.mock.patch`              |
| Only testing the happy path                     | Production bugs in error paths aren't caught                | Explicitly write tests for invalid input and network failures     |
| Testing implementation details (SQL query text) | Tests break on refactor                                     | Test the tool's _contract_: input → output shape, not internals   |
| Forgetting to test the schema                   | Model can't fill args correctly; only caught in agent tests | Test `.args_schema.model_json_schema()` for required descriptions |

---

## Mini Summary

- Test every tool with `.invoke(args)` directly — no agent, no LLM, no environment variables needed
- Mock all external I/O (HTTP, DB, filesystem) so tests are fast, deterministic, and offline
- Test three paths: happy path, validation failure, infrastructure failure
- Test the schema: verify that required fields have meaningful `Field(description=...)` values
- A tool that passes its isolated test suite can be trusted in the agent

---

[← Tool Exception Patterns](03-tool-exception-patterns.md) | [Next → Tool Registry Pattern](05-tool-registry-pattern.md)
