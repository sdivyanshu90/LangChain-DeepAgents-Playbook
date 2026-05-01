[← Binding Tools](02-binding-tools.md) | [Next → Tool Error Handling](04-tool-error-handling.md)

---

# 03 — Tool Message Parsing

## Why Strict Message Ordering Exists

The OpenAI chat API is stateless. Every call sends the full conversation history.
The API validates that each `tool_calls` entry in an `AIMessage` has a matching
`ToolMessage` response _before_ accepting the next `AIMessage`. This strict pairing
prevents the model from "forgetting" that it asked for a tool result and inventing
a fake answer instead.

Understanding this constraint is essential — break the ordering rule and you get a
cryptic 400 error that doesn't tell you what actually went wrong.

---

## Real-World Analogy

Imagine you're a project manager who just sent five RFQs (Requests for Quotation)
to five vendors. You cannot close the procurement until all five vendors have replied.
If you tried to sign a contract before hearing back from vendor #3, accounts payable
would reject it.

The model is the project manager. Each `ToolCall` is an RFQ. Each `ToolMessage` is a
vendor reply. The API is accounts payable — it won't accept the next decision until
all outstanding RFQs have a matching reply.

---

## `ToolCall` vs `ToolMessage` — Anatomy

```
ToolCall (inside AIMessage.tool_calls)       ToolMessage
─────────────────────────────────────────    ─────────────────────────────────────────
name: str          ← which function          content: str        ← the return value
args: dict         ← parsed arguments        tool_call_id: str   ← MUST match ToolCall.id
id: str            ← unique call ID          type: "tool"        ← always "tool"
type: "tool_call"  ← always "tool_call"
```

The `id` field is the contract. `ToolCall.id` ↔ `ToolMessage.tool_call_id` must match exactly.

---

## The Complete Two-Step Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOOL CALL CYCLE                                      │
│                                                                             │
│  ┌───────────┐  invoke   ┌──────────────────────┐                          │
│  │           │──────────▶│ LLM                  │                          │
│  │ App Code  │           │                      │                          │
│  │           │           │  Emits AIMessage with│                          │
│  │           │◀──────────│  tool_calls=[         │                          │
│  │           │  response │    ToolCall(id="c1"), │                          │
│  └───────────┘           │    ToolCall(id="c2")  │                          │
│       │                  │  ]                   │                          │
│       │                  └──────────────────────┘                          │
│       │ (app executes each tool)                                            │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────┐                          │
│  │  messages = [                                │                          │
│  │    HumanMessage("..."),                      │  ← original user query   │
│  │    AIMessage(tool_calls=[c1, c2]),           │  ← step 1 response       │
│  │    ToolMessage(tool_call_id="c1", ...),      │  ← result for c1         │
│  │    ToolMessage(tool_call_id="c2", ...),      │  ← result for c2         │
│  │  ]                                           │                          │
│  └──────────────────────────────────────────────┘                          │
│       │ invoke again                                                        │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────┐                          │
│  │  AIMessage(content="Final answer to user")   │                          │
│  └──────────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full Working Code Example

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ── Tool definitions ───────────────────────────────────────────────────────────
@tool
def get_stock_price(ticker: str) -> str:
    """Return the current stock price for a ticker symbol (e.g. AAPL, TSLA)."""
    # Stub implementation
    prices = {"AAPL": "$182.50", "TSLA": "$245.10", "GOOGL": "$175.20"}
    return prices.get(ticker.upper(), f"Ticker {ticker} not found.")

@tool
def get_company_news(ticker: str) -> str:
    """Fetch the latest news headline for a company given its ticker symbol."""
    news = {
        "AAPL": "Apple announces Vision Pro 2 with 40% performance improvement.",
        "TSLA": "Tesla Q1 deliveries beat analyst estimates by 8%.",
    }
    return news.get(ticker.upper(), "No recent news found.")

# ── Model setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_stock_price, get_company_news]
llm_with_tools = llm.bind_tools(tools)

# ── Tool registry for dispatch (maps name → callable) ─────────────────────────
tool_map = {t.name: t for t in tools}

# ── Run the full loop ─────────────────────────────────────────────────────────
def run_tool_loop(user_input: str, max_iterations: int = 5) -> str:
    """
    Execute the tool-call loop until the model produces a final text answer.
    Returns the final AIMessage content.
    """
    messages = [HumanMessage(content=user_input)]

    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls, the model has produced its final answer
        if not response.tool_calls:
            return response.content

        # Execute every tool call and collect ToolMessages
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]

            if tool_name not in tool_map:
                # Return an error ToolMessage for unknown tools
                tool_result = f"Error: unknown tool '{tool_name}'"
            else:
                tool_result = tool_map[tool_name].invoke(tool_args)

            # Append ToolMessage with matching tool_call_id
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                )
            )

    return "Max iterations reached without a final answer."

# ── Test it ───────────────────────────────────────────────────────────────────
answer = run_tool_loop("What's the current price of AAPL and any recent news?")
print(answer)
# "Apple (AAPL) is currently trading at $182.50.
#  The latest news: Apple announces Vision Pro 2 with 40% performance improvement."
```

---

## Handling Invalid JSON in `arguments`

Although rare with capable models, the `arguments` field (raw JSON string from the API)
can be malformed. LangChain catches this and surfaces it in `invalid_tool_calls`:

```python
from langchain_core.messages import AIMessage

# Normal case — properly parsed:
response = llm_with_tools.invoke(messages)
print(response.tool_calls)        # list of valid ToolCall dicts
print(response.invalid_tool_calls)  # [] when all is well

# If the model produced broken JSON, LangChain puts it here:
# response.invalid_tool_calls = [
#   {'name': 'get_stock_price', 'args': '{"ticker": ', 'id': 'call_xyz', 'error': '...'}
# ]

# Defensive check:
if response.invalid_tool_calls:
    for bad_call in response.invalid_tool_calls:
        print(f"Invalid tool call: {bad_call['name']}: {bad_call['error']}")
        # Return an error ToolMessage so the model can self-correct:
        messages.append(
            ToolMessage(
                content=f"Tool call failed to parse: {bad_call['error']}. Please retry with valid JSON arguments.",
                tool_call_id=bad_call["id"],
            )
        )
```

---

## `ToolMessage` Content Types

`ToolMessage.content` can be a plain string OR a list of content blocks
(for multimodal tool results — e.g., an image from a screenshot tool):

```python
# Plain string (most common):
ToolMessage(content="{'temp': 18, 'condition': 'cloudy'}", tool_call_id="c1")

# Structured content block (for multimodal):
ToolMessage(
    content=[{"type": "text", "text": "Screenshot captured."}, {"type": "image_url", "image_url": "..."}],
    tool_call_id="c1",
)
```

Most tool implementations return a plain string or a JSON-stringified dict.

---

## Why You Must Not Skip the `AIMessage` in History

This is a common mistake when developers try to "clean up" the conversation history:

```python
# ❌ WRONG — ToolMessage without preceding AIMessage(tool_calls)
messages = [
    HumanMessage("What is AAPL price?"),
    # AIMessage omitted to "save tokens"
    ToolMessage(content="$182.50", tool_call_id="c1"),  # API rejects this
]

# ✅ CORRECT — AIMessage must precede all its ToolMessages
messages = [
    HumanMessage("What is AAPL price?"),
    AIMessage(content="", tool_calls=[{"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "c1"}]),
    ToolMessage(content="$182.50", tool_call_id="c1"),
]
```

If you want to compress history, use a summary strategy (Module 2.1) after the tool cycle
completes — never delete the AIMessage mid-cycle.

---

## Common Pitfalls

| Pitfall                                   | Symptom                                        | Fix                                                            |
| ----------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| Wrong `tool_call_id` in `ToolMessage`     | API 400: "no matching tool call id"            | Copy `tool_call["id"]` directly, never hardcode                |
| Forgetting to handle `invalid_tool_calls` | Silent failure; model hallucinates next answer | Always check `response.invalid_tool_calls`                     |
| One `ToolMessage` for two parallel calls  | API 400: missing tool result                   | Loop over ALL `response.tool_calls`, append ALL `ToolMessage`s |
| Stringifying tool output as `str(dict)`   | Model can't parse `"{'a': 1}"` (single quotes) | Use `json.dumps(result)` for dict outputs                      |
| Max iterations never set                  | Infinite loop if model keeps calling tools     | Add `max_iterations` guard in your loop                        |

---

## Mini Summary

- A `ToolCall` lives inside `AIMessage.tool_calls`; a `ToolMessage` is the app's reply
- The `id` field bridges them: `ToolCall.id` must equal `ToolMessage.tool_call_id`
- The message ordering rule is: `AIMessage(tool_calls=[...])` → all matching `ToolMessage`s → next LLM call
- Parallel tool calls are supported — you must return one `ToolMessage` per call before the next LLM turn
- Invalid JSON in `arguments` surfaces in `AIMessage.invalid_tool_calls` — handle it to allow self-correction

---

[← Binding Tools](02-binding-tools.md) | [Next → Tool Error Handling](04-tool-error-handling.md)
