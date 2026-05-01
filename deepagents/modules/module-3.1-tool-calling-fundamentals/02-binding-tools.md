[← How Tool Calling Works](01-how-tool-calling-works.md) | [Next → Tool Message Parsing](03-tool-message-parsing.md)

---

# 02 — Binding Tools to a Model

## Why `.bind_tools()` Exists

Without `.bind_tools()`, you would have to manually serialize each tool to a JSON schema,
append it to every API call, and parse the `tool_calls` array in every response yourself.
The entire point of `.bind_tools()` is to make tool configuration a one-time setup so the
rest of your code can treat the model as a normal Runnable.

---

## Real-World Analogy

Think of a contractor who has a toolbox. Before any job begins, you agree on which tools
they're allowed to use on this project: power drill, tile saw, paint roller — not the
jackhammer. This agreement (`bind_tools`) is made once. During the job, the contractor
picks whichever permitted tool fits the task. You don't re-negotiate tool access every hour.

`.bind_tools()` is that agreement. You make it once; every subsequent invocation respects it.

---

## Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information. query is a plain-language question."""
    return f"[STUB] Search results for: {query}"

@tool
def read_file(path: str) -> str:
    """Read a file from the local filesystem and return its contents."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind both tools — returns a new Runnable, does NOT mutate llm
llm_with_tools = llm.bind_tools([search_web, read_file])

# llm is unchanged; llm_with_tools always sends tool schemas in every API call
response = llm_with_tools.invoke("What is the latest news about LangChain?")
print(response.tool_calls)
# [{'name': 'search_web', 'args': {'query': 'latest news about LangChain'}, 'id': '...'}]
```

---

## `tool_choice` — Controlling When the Model Calls Tools

The `tool_choice` parameter is injected into the API request. It controls whether the
model _may_ call tools, _must_ call a specific tool, or _must_ call _some_ tool.

```python
# ── "auto" (default) ──────────────────────────────────────────────────────────
# The model decides: call a tool OR answer directly.
llm_auto = llm.bind_tools([search_web, read_file], tool_choice="auto")

# ── "any" ─────────────────────────────────────────────────────────────────────
# The model MUST call one of the provided tools. Will not answer directly.
# Useful when you've already decided a tool is needed (e.g., always search first).
llm_must_use_tool = llm.bind_tools([search_web, read_file], tool_choice="any")

# ── "required" (OpenAI alias for "any") ───────────────────────────────────────
llm_required = llm.bind_tools([search_web, read_file], tool_choice="required")

# ── Specific tool name ─────────────────────────────────────────────────────────
# Force the model to call EXACTLY this tool. Ignores all others.
# Useful for structured extraction: "always call extract_entities, never skip it."
llm_forced = llm.bind_tools([search_web, read_file], tool_choice="search_web")

# ── "none" ────────────────────────────────────────────────────────────────────
# Completely suppresses tool calling. The model will answer directly even if
# tools are defined. Used when you want the final synthesis step to skip tools.
llm_no_tools = llm.bind_tools([search_web, read_file], tool_choice="none")
```

### Decision Guide

```
┌─────────────────────────────────────────────────────────┐
│          Choosing tool_choice                           │
├──────────────────┬──────────────────────────────────────┤
│ "auto"           │ Normal agent — model decides          │
│ "any"/"required" │ Always want a tool (extraction step)  │
│ "tool_name"      │ Structured extraction, one specific   │
│                  │ schema to fill                        │
│ "none"           │ Synthesis step — no more tools needed │
└──────────────────┴──────────────────────────────────────┘
```

---

## Parallel Tool Calls

OpenAI models can emit **multiple** `ToolCall` objects in a single response when
`tool_choice="auto"` and the user query benefits from parallel action:

```python
from langchain_core.messages import HumanMessage, ToolMessage

# Query that naturally benefits from parallel calls:
messages = [HumanMessage("What's the weather in Paris AND London?")]
response = llm_with_tools.invoke(messages)

print(len(response.tool_calls))  # 2

# You MUST execute ALL tool calls and return ALL ToolMessages before the next LLM call.
# Returning only one ToolMessage while omitting the other causes an API error.

tool_results = []
for tc in response.tool_calls:
    result = get_weather.invoke(tc["args"])   # hypothetical weather tool
    tool_results.append(
        ToolMessage(content=result, tool_call_id=tc["id"])
    )

# Feed all results back together:
final = llm_with_tools.invoke([*messages, response, *tool_results])
print(final.content)
```

### Disabling Parallel Calls

Some models support disabling this. Pass `parallel_tool_calls=False` if your downstream
system can only handle sequential results:

```python
llm_sequential = llm.bind_tools(
    [search_web, read_file],
    parallel_tool_calls=False,  # supported by gpt-4o family
)
```

---

## Inspecting What Was Bound

After calling `.bind_tools()`, you can inspect the bound schemas:

```python
# The Runnable's kwargs include the serialized tools:
bound_kwargs = llm_with_tools.kwargs
print(bound_kwargs.keys())
# dict_keys(['tools', 'tool_choice'])

# See the full schema for each tool:
import json
for t in bound_kwargs["tools"]:
    print(t["function"]["name"])
    print(json.dumps(t["function"]["parameters"], indent=2))
    print()
```

You can also inspect the original tool objects:

```python
# The @tool decorator gives each tool a .name and .description:
print(search_web.name)         # "search_web"
print(search_web.description)  # "Search the web for current information..."

# And a .args_schema that matches what gets serialized:
print(search_web.args_schema.model_json_schema())
```

---

## Binding Tools at Runtime vs at Definition Time

Sometimes you want to decide _which_ tools to bind based on the user or the task:

```python
from langchain_core.tools import tool

# A library of available tools:
ALL_TOOLS = {
    "search_web": search_web,
    "read_file": read_file,
    "write_file": write_file,    # hypothetical
    "send_email": send_email,    # hypothetical
}

def get_agent_for_role(role: str):
    """Return an LLM bound only to the tools appropriate for this role."""
    role_tools = {
        "researcher": ["search_web", "read_file"],
        "writer": ["read_file", "write_file"],
        "manager": ["search_web", "send_email"],
    }
    tools = [ALL_TOOLS[name] for name in role_tools.get(role, [])]
    return llm.bind_tools(tools, tool_choice="auto")

researcher_llm = get_agent_for_role("researcher")
```

---

## LCEL Chains with Bound Tools

`.bind_tools()` returns a plain `Runnable`, so it composes naturally:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use tools when needed."),
    ("human", "{input}"),
])

# Build chain: prompt → model-with-tools
chain = prompt | llm_with_tools

response = chain.invoke({"input": "Find the latest Python 3.13 release notes."})
```

---

## Common Pitfalls

| Pitfall                                                     | Symptom                                                | Fix                                                                           |
| ----------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Binding the same tool twice                                 | Duplicate schema; model may call it twice              | Use `{t.name: t for t in tools}.values()` to deduplicate                      |
| Using `tool_choice="required"` at the final synthesis step  | Model tries to call a tool even when it has the answer | Switch to `tool_choice="none"` for synthesis                                  |
| Large docstrings on many tools                              | Prompt bloat, context limit hits                       | Keep descriptions ≤ 100 characters; put examples in the system prompt instead |
| Forgetting `parallel_tool_calls=False` in rate-limited APIs | Burst of simultaneous API calls causes 429 errors      | Disable parallel calls or add retry/backoff                                   |
| Mutating `llm` instead of binding to a new variable         | All subsequent calls also send unwanted tools          | `.bind_tools()` is non-mutating — always assign to a new variable             |

---

## Mini Summary

- `.bind_tools(tools)` serializes tool schemas and attaches them to every API call; it returns a new Runnable
- `tool_choice` controls whether the model _may_ call tools (`"auto"`), _must_ call one (`"any"`/`"required"`), or is forced to skip tools (`"none"`)
- A single response may contain multiple `ToolCall` objects — always loop over `ai_msg.tool_calls`, never assume just one
- You can inspect bound schemas via `llm_with_tools.kwargs["tools"]`
- `.bind_tools()` composes naturally in LCEL chains

---

[← How Tool Calling Works](01-how-tool-calling-works.md) | [Next → Tool Message Parsing](03-tool-message-parsing.md)
