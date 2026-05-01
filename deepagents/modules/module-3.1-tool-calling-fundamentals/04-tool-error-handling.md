[← Tool Message Parsing](03-tool-message-parsing.md) | [Next → When to Use Tools](05-when-to-use-tools.md)

---

# 04 — Tool Error Handling

## Why Tools Fail (and Why That's Fine)

A tool that can never fail is either too simple or lying. Real tools call APIs that go down,
query databases that timeout, parse inputs that violate assumptions, and hit rate limits.
The failure mode that kills agents is not _tools failing_ — it's **tools failing silently**
(returning empty, crashing the process, or corrupting the message history).

Robust tool error handling ensures that when a tool fails, the model knows _why_ it failed
and can decide what to do next: retry with different arguments, try a fallback tool, or
report the failure to the user gracefully.

---

## Real-World Analogy

When a contractor can't complete a task — "the part is out of stock" — they don't disappear.
They call you and explain: "Part #A14 is backordered; I can use Part #B22 as a substitute
for 10% more cost, or we wait two weeks. What do you prefer?"

That's what a properly error-handling tool does: it explains the failure clearly so the
decision-maker (the model, the user) can respond intelligently.

---

## Three Ways Tool Errors Surface

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Tool raises Python exception                                            │
│                                                                          │
│  Case 1: UNHANDLED                                                       │
│    Exception propagates → process crashes OR LangGraph node crashes      │
│    ❌ The ToolMessage is never added to the message list                  │
│    ❌ The model never learns what happened                                │
│                                                                          │
│  Case 2: handle_tool_error=True (LangChain catches it)                   │
│    Exception message is caught → turned into a ToolMessage(content=err)  │
│    ✅ The model sees the error and can reason about it                    │
│                                                                          │
│  Case 3: "Safe tool" pattern (never raises, always returns dict)         │
│    Tool catches its own exceptions and returns {"error": "...", ...}      │
│    ✅ Full control over error structure; works in any framework           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## `ToolException` — The Standard Signal

`ToolException` is LangChain's explicit "expected failure" exception.
Use it when you want the model to receive a structured error message:

```python
from langchain_core.tools import tool, ToolException

@tool
def fetch_user_profile(user_id: str) -> dict:
    """Fetch a user's profile from the database by user_id."""
    if not user_id.isdigit():
        # This is an expected, recoverable failure — tell the model how to fix it
        raise ToolException(
            f"Invalid user_id '{user_id}'. user_id must be a numeric string, e.g. '12345'."
        )
    # Stub: real implementation queries DB
    return {"id": user_id, "name": "Alice", "plan": "pro"}
```

---

## `handle_tool_error` on Individual Tools

When you use `@tool` with `handle_tool_error=True`, LangChain wraps the function in a
try/except block. If a `ToolException` (or any exception) is raised, it converts the
exception message into a `ToolMessage` automatically.

```python
from langchain_core.tools import tool, ToolException

@tool(handle_tool_error=True)
def fetch_user_profile(user_id: str) -> str:
    """Fetch a user profile from the database."""
    if not user_id.isdigit():
        raise ToolException(
            f"Invalid user_id '{user_id}'. Must be numeric, e.g. '12345'."
        )
    return f"User {user_id}: Alice (pro plan)"
```

### Custom Error Handler

Pass a callable instead of `True` to format the error message yourself:

```python
def format_error(error: Exception) -> str:
    """Turn any exception into a model-friendly error string."""
    if isinstance(error, ToolException):
        return f"[TOOL ERROR] {str(error)}\nPlease correct your input and retry."
    return f"[UNEXPECTED ERROR] {type(error).__name__}: {str(error)}\nThis may be a transient issue — you can retry."

@tool(handle_tool_error=format_error)
def search_database(query: str) -> str:
    """Search the internal knowledge base."""
    if len(query.strip()) < 3:
        raise ToolException("Query too short. Provide at least 3 characters.")
    # Simulate a database call
    return f"Found 5 results for '{query}'."
```

---

## The Safe Tool Pattern

The `handle_tool_error` flag requires the `@tool` decorator to be aware of errors.
In some frameworks (like custom LangGraph nodes) you invoke tools directly with `.invoke()`.
The **safe tool pattern** wraps all error handling _inside the tool function itself_,
making it framework-agnostic:

```python
import json
from typing import Any
from langchain_core.tools import tool

@tool
def safe_web_search(query: str) -> str:
    """
    Search the web for current information.
    Always returns a JSON string with either 'results' or 'error'.
    Never raises an exception.
    """
    try:
        # Stub: replace with actual HTTP call
        if not query.strip():
            return json.dumps({"error": "Empty query provided.", "results": None})

        results = [f"Result snippet for: {query}"]   # mock
        return json.dumps({"error": None, "results": results})

    except Exception as e:
        # Catch all unexpected errors; return structured failure
        return json.dumps({
            "error": f"{type(e).__name__}: {str(e)}",
            "results": None,
        })
```

The model receives a consistent JSON structure regardless of what went wrong, and can
extract `error` or `results` reliably.

---

## Handling Errors in a LangGraph Tool Node

In a LangGraph agent, tool execution typically happens in a dedicated node.
Here's a production-ready implementation:

```python
import json
from typing import Any
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import BaseTool, ToolException

def build_tool_executor_node(tools: list[BaseTool]):
    """
    Return a LangGraph node that executes tool calls from the last AIMessage.
    Produces one ToolMessage per ToolCall, including graceful error messages.
    """
    tool_map = {t.name: t for t in tools}

    def tool_node(state: dict) -> dict:
        # Find the last AIMessage with tool calls
        messages = state["messages"]
        last_ai = messages[-1]

        if not isinstance(last_ai, AIMessage) or not last_ai.tool_calls:
            raise ValueError("tool_node called but last message has no tool_calls.")

        tool_messages = []
        for tc in last_ai.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id   = tc["id"]

            tool = tool_map.get(tool_name)
            if tool is None:
                # Unknown tool — send error ToolMessage
                content = json.dumps({
                    "error": f"Tool '{tool_name}' is not available.",
                    "available_tools": list(tool_map.keys()),
                })
            else:
                try:
                    result = tool.invoke(tool_args)
                    # Normalise: tools may return str, dict, list — serialise all to str
                    content = result if isinstance(result, str) else json.dumps(result)
                except ToolException as e:
                    content = json.dumps({
                        "error": str(e),
                        "tool": tool_name,
                        "hint": "Check argument types and values.",
                    })
                except Exception as e:
                    # Unexpected error — log for debugging, don't crash the graph
                    content = json.dumps({
                        "error": f"Unexpected error in tool '{tool_name}': {type(e).__name__}: {str(e)}",
                        "tool": tool_name,
                    })

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tool_id)
            )

        return {"messages": tool_messages}

    return tool_node
```

---

## Graceful Degradation — Providing Partial Results

Sometimes a tool can give _some_ useful output even under failure conditions.
Return partial results with an explicit degradation signal:

```python
@tool
def batch_fetch_prices(tickers: list[str]) -> str:
    """
    Fetch current prices for a list of ticker symbols.
    Returns results for all tickers that succeeded; marks failures explicitly.
    """
    results = {}
    failed = []
    for ticker in tickers:
        try:
            # Stub: real API call here
            results[ticker] = f"${100 + len(ticker)}.00"   # mock
        except Exception as e:
            failed.append({"ticker": ticker, "error": str(e)})

    return json.dumps({
        "results": results,
        "failed": failed,
        "partial": len(failed) > 0,
    })
```

The model can see `"partial": true` and decide to retry the failed tickers or inform the user.

---

## Common Pitfalls

| Pitfall                                                | Symptom                                              | Fix                                                      |
| ------------------------------------------------------ | ---------------------------------------------------- | -------------------------------------------------------- |
| Letting exceptions propagate uncaught                  | LangGraph node crashes; entire run fails             | Always catch in the tool node OR use `handle_tool_error` |
| Raising generic `Exception` instead of `ToolException` | Model gets a Python traceback it can't act on        | Raise `ToolException` with a human-readable message      |
| Returning `None` on failure                            | `str(None) == "None"` in ToolMessage; model confused | Return a JSON string with an `"error"` key               |
| Swallowing errors with empty string return             | Model thinks the tool succeeded with empty output    | Always distinguish empty result from failure             |
| Infinite retry when tool always fails                  | Agent loops until max_iterations                     | Detect repeated failures in state and route to fallback  |

---

## Mini Summary

- Tools fail predictably in production: handle errors deliberately, not by accident
- `ToolException` is the semantic "expected failure" signal — raise it with a clear message
- `handle_tool_error=True` on `@tool` catches exceptions and converts them to `ToolMessage` content automatically
- The **safe tool pattern** — catching internally and returning `{"error": ...}` — works anywhere, regardless of framework
- In a LangGraph tool node, wrap every `.invoke()` call in try/except to prevent a single tool failure from crashing the whole run
- Partial results with an explicit `"partial": true` flag are more useful to the model than hard failures

---

[← Tool Message Parsing](03-tool-message-parsing.md) | [Next → When to Use Tools](05-when-to-use-tools.md)
