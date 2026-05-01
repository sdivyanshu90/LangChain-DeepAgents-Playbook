[← The @tool Decorator](01-tool-decorator.md) | [Next → Tool Exception Patterns](03-tool-exception-patterns.md)

---

# 02 — Async Tools

## Why Async Tools Matter

A synchronous tool call blocks the event loop for as long as the tool takes to run.
For a single-tool agent this is fine. For a multi-agent system with parallel tool calls,
where multiple agents are running concurrently in a single Python process, blocking the
event loop means every _other_ agent is frozen while one agent waits for an HTTP response.

Async tools let multiple tool calls and agent steps happen truly concurrently, without
spawning OS threads, without the GIL bottleneck of CPU-bound threads, and without the
overhead of multiprocessing.

---

## Real-World Analogy

A synchronous waiter takes your order, walks to the kitchen, stands there until the food
is ready, then walks back. While they wait, every other table is ignored.

An async waiter takes your order, submits it to the kitchen, and immediately serves
another table. When the kitchen calls "order up!", they return to deliver your food.
Same person, same kitchen — but dramatically higher throughput.

LangChain's async tool invocation (`ainvoke`) is the async waiter.

---

## Sync Tool vs Async Tool

```
Sync tool                              Async tool
──────────────────────────────────     ──────────────────────────────────────
@tool                                  @tool
def search_web(query: str) -> str:     async def search_web(query: str) -> str:
    resp = requests.get(url)               async with aiohttp.ClientSession() as s:
    return resp.text                           resp = await s.get(url)
                                               return await resp.text()

Invocation:                            Invocation:
tool.invoke({"query": "..."})          await tool.ainvoke({"query": "..."})

Blocks event loop: YES                 Blocks event loop: NO
Suitable for LangGraph: marginal       Suitable for LangGraph: YES
```

---

## Defining an Async Tool with `@tool`

```python
import aiohttp
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WebSearchInput(BaseModel):
    query: str = Field(
        description="Search query. Be specific; avoid single words.",
        min_length=3,
    )
    num_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of search result snippets to return.",
    )

@tool("search_web_async", args_schema=WebSearchInput)
async def search_web_async(query: str, num_results: int = 3) -> str:
    """
    Search the web for current information using the Brave Search API.
    Returns a JSON array of {title, url, snippet} objects.
    Use this for any question about current events, recent releases, or live data.
    Requires BRAVE_API_KEY environment variable.
    """
    import os
    api_key = os.getenv("BRAVE_API_KEY", "")
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": num_results}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        }
        for r in data.get("web", {}).get("results", [])[:num_results]
    ]
    return json.dumps(results, ensure_ascii=False)
```

---

## Coroutine vs Regular Function in Tool Binding

When you call `llm.bind_tools([tool])`, LangChain doesn't care whether `tool` is sync or async.
The schema generation is the same. The difference appears at _invocation time_:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([search_web_async])

# Sync invocation — LangChain runs the coroutine in a new event loop:
response = llm_with_tools.invoke([HumanMessage("Latest Python 3.13 features?")])

# Async invocation — runs naturally in an existing event loop:
# response = await llm_with_tools.ainvoke([HumanMessage("...")])
```

When LangChain's `.invoke()` encounters an async tool, it runs it with `asyncio.run()` internally.
This is fine in scripts but **will raise an error if called from within an already-running event loop**
(e.g., inside a Jupyter notebook or inside a LangGraph async graph).

**Rule:** Use `await tool.ainvoke(args)` whenever you're already in async context.

---

## Parallel Async Tool Execution

The real win of async tools comes when executing multiple tool calls concurrently:

```python
import asyncio
from langchain_core.messages import AIMessage, ToolMessage

async def execute_tool_calls_parallel(
    tool_calls: list,
    tool_map: dict,
) -> list[ToolMessage]:
    """
    Execute all tool calls from an AIMessage concurrently.
    Returns one ToolMessage per tool call.
    """
    async def run_one(tc: dict) -> ToolMessage:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id   = tc["id"]

        tool = tool_map.get(tool_name)
        if tool is None:
            content = json.dumps({"error": f"Unknown tool '{tool_name}'"})
        else:
            try:
                # Use ainvoke if the tool is async, invoke otherwise
                if asyncio.iscoroutinefunction(tool.func):
                    result = await tool.ainvoke(tool_args)
                else:
                    # Run blocking tool in a thread pool to avoid blocking the loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, tool.invoke, tool_args)
                content = result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
                content = json.dumps({"error": str(e)})

        return ToolMessage(content=content, tool_call_id=tool_id)

    # Run all tool calls concurrently with asyncio.gather
    return list(await asyncio.gather(*[run_one(tc) for tc in tool_calls]))


# Usage in a LangGraph-style async agent:
async def async_tool_node(state: dict) -> dict:
    messages = state["messages"]
    last_ai = messages[-1]
    tool_messages = await execute_tool_calls_parallel(last_ai.tool_calls, TOOL_MAP)
    return {"messages": tool_messages}
```

---

## The Asyncio Event Loop in LangGraph

LangGraph's async graph is driven by `asyncio`. When you compile a graph and call
`await graph.ainvoke(...)`, every node runs in the same event loop. This means:

- Async tool nodes run natively: `await tool.ainvoke(args)` just works
- Sync tool nodes are automatically wrapped in `asyncio.to_thread()` by LangGraph
- You should NOT use `asyncio.run()` inside a LangGraph node — you're already inside a running loop

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# Async node — correct pattern inside LangGraph async graph:
async def async_tool_node(state: AgentState) -> dict:
    last_ai = state["messages"][-1]
    results = await execute_tool_calls_parallel(last_ai.tool_calls, TOOL_MAP)
    return {"messages": results}

# Sync node — also fine in async graph (LangGraph wraps it):
def sync_agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", sync_agent_node)
graph.add_node("tools", async_tool_node)
# ...
```

---

## Running Async Tools Outside LangGraph

For scripts or tests that aren't inside a running event loop:

```python
import asyncio

# In a script:
result = asyncio.run(search_web_async.ainvoke({"query": "LangGraph 0.3 release"}))

# In a Jupyter notebook (event loop already running):
# Use nest_asyncio to allow nested loops:
import nest_asyncio
nest_asyncio.apply()
result = asyncio.run(search_web_async.ainvoke({"query": "LangGraph 0.3 release"}))
# OR use await directly in a notebook cell:
# result = await search_web_async.ainvoke({"query": "..."})
```

---

## Performance Comparison — Sync vs Async

```
Scenario: 4 tool calls, each takes 500ms (HTTP request)

Sync sequential:    500 + 500 + 500 + 500 = 2000ms total
Async concurrent:   max(500, 500, 500, 500) = 500ms total  (4× faster)

For N=10 tools each taking 1s:
Sync:   10 seconds
Async:  ~1 second
```

The speedup grows linearly with the number of concurrent tool calls.
This matters in research agents that fan out to multiple APIs simultaneously.

---

## Common Pitfalls

| Pitfall                                                                  | Symptom                                             | Fix                                                                             |
| ------------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------- |
| `asyncio.run()` inside a running event loop                              | `RuntimeError: This event loop is already running`  | Use `await tool.ainvoke()` instead                                              |
| Mixing sync `requests` inside async tool                                 | Blocks the event loop; kills concurrency gains      | Replace with `aiohttp`, `httpx.AsyncClient`, or `asyncio.to_thread`             |
| Forgetting `await` on `ainvoke`                                          | Returns a coroutine object, not the result          | Always `await tool.ainvoke(args)` in async context                              |
| Using sync tool in LangGraph async graph without `to_thread`             | Event loop blocked during tool execution            | LangGraph wraps sync nodes automatically; still prefer async for IO-bound tools |
| Not checking `asyncio.iscoroutinefunction(tool.func)` in mixed tool sets | Calling `await` on a sync result raises `TypeError` | Use the `iscoroutinefunction` check before awaiting                             |

---

## Mini Summary

- `async def` tools don't block the event loop; they're essential for concurrent agents
- Use `await tool.ainvoke(args)` inside async contexts (LangGraph, `async def` functions)
- `asyncio.gather()` runs multiple tool calls concurrently — 4× speedup for 4 parallel calls
- Inside a LangGraph async graph, you're already in a running event loop — never call `asyncio.run()` there
- Sync tools can coexist with async tools; use `asyncio.to_thread(tool.invoke, args)` to keep sync tools non-blocking

---

[← The @tool Decorator](01-tool-decorator.md) | [Next → Tool Exception Patterns](03-tool-exception-patterns.md)
