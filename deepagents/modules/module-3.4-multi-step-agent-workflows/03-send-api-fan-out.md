[← Human-in-the-Loop](02-human-in-the-loop.md) | [Next → Retry and Fallback](04-retry-and-fallback.md)

---

# 03 — Send API Fan-Out

## Why Dynamic Parallelism Exists

`RunnableParallel` in LCEL runs a _fixed_ set of tasks in parallel — you know at graph
design time exactly how many and which tasks will run.

A research agent that receives "Compare the top 5 competitors of company X" doesn't know
at design time that there are 5 competitors. It only knows this at runtime, after a
planning step identifies them.

The **Send API** solves this: it lets a node dynamically create an arbitrary number of
parallel graph invocations at runtime, one per item in a list.

---

## Real-World Analogy

A law firm receives a discovery request with 200 documents. The senior partner assigns
each document to a different associate for independent review. All 200 reviews happen
simultaneously. When every associate has filed their summary, the partner synthesises
the findings.

The Send API is the senior partner's assignment system: dynamic number of parallel
workers, independent execution, results collected and synthesised at the end.

---

## How `Send` Works

```python
from langgraph.types import Send

# In a "map" node, return a list of Send objects instead of a state update:
def map_node(state: State) -> list[Send]:
    """
    For each item in state["items"], send it to a worker node for independent processing.
    Each Send creates an isolated subgraph invocation with its own state.
    """
    return [
        Send("worker_node", {"item": item, "context": state["context"]})
        for item in state["items"]
    ]
```

Each `Send("node_name", state_dict)` creates one independent invocation of `"node_name"`.
All invocations run concurrently. Their results are collected by the reducer on the
corresponding field.

---

## Complete Example — Research Fan-Out

```python
# research_fanout.py
from typing import TypedDict, Annotated
import operator
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════
class ResearchState(TypedDict):
    # Main orchestrator state
    topic:       str
    sub_queries: list[str]                              # populated by planner
    results:     Annotated[list[dict], operator.add]   # collected from all workers
    final_report: str

class WorkerState(TypedDict):
    # Each worker gets this isolated state
    item:    str   # the sub-query to research
    context: str   # the parent topic, for context
    results: Annotated[list[dict], operator.add]

# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_synth = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ═══════════════════════════════════════════════════════════════════
# NODE 1: Planner — generates sub-queries
# ═══════════════════════════════════════════════════════════════════
def planner_node(state: ResearchState) -> dict:
    """
    Break the research topic into focused sub-queries.
    Returns a list of sub-queries to be researched in parallel.
    """
    system = SystemMessage(
        "You are a research planner. Given a topic, generate 3-5 focused sub-queries "
        "that together would cover the topic comprehensively.\n"
        "Respond with a JSON array of strings: [\"query1\", \"query2\", ...]"
    )
    response = llm.invoke([system, HumanMessage(state["topic"])])
    try:
        queries = json.loads(response.content)
        if not isinstance(queries, list):
            queries = [state["topic"]]  # fallback
    except json.JSONDecodeError:
        queries = [state["topic"]]      # fallback

    return {"sub_queries": queries}

# ═══════════════════════════════════════════════════════════════════
# FAN-OUT NODE: Create parallel Send objects
# ═══════════════════════════════════════════════════════════════════
def fan_out_node(state: ResearchState) -> list[Send]:
    """
    For each sub-query, send an independent research task to the worker node.
    All workers run concurrently.
    """
    return [
        Send(
            "research_worker",           # the node to run
            {                            # the WorkerState for this invocation
                "item":    query,
                "context": state["topic"],
                "results": [],
            }
        )
        for query in state["sub_queries"]
    ]

# ═══════════════════════════════════════════════════════════════════
# NODE 2: Worker — researches one sub-query
# ═══════════════════════════════════════════════════════════════════
def research_worker(state: WorkerState) -> dict:
    """
    Research a single sub-query independently.
    Each Send invocation calls this node with isolated WorkerState.
    """
    system = SystemMessage(
        f"You are a research assistant. Context: {state['context']}\n"
        "Research the given sub-query thoroughly and provide a concise 2-3 paragraph summary "
        "with key facts, figures, and insights."
    )
    response = llm.invoke([system, HumanMessage(state["item"])])
    result = {
        "query":   state["item"],
        "summary": response.content,
    }
    return {"results": [result]}   # operator.add appends this to the parent's results list

# ═══════════════════════════════════════════════════════════════════
# NODE 3: Synthesiser — merges all results
# ═══════════════════════════════════════════════════════════════════
def synthesise_node(state: ResearchState) -> dict:
    """
    Synthesise all parallel research results into a cohesive report.
    Runs after ALL workers complete.
    """
    # Format collected results for the synthesiser prompt
    summaries = "\n\n".join(
        f"### Sub-query: {r['query']}\n{r['summary']}"
        for r in state["results"]
    )

    system = SystemMessage(
        "You are a research editor. Given the following research summaries on sub-aspects "
        "of a topic, write a coherent, well-structured executive summary covering the full topic. "
        "Do not just concatenate the summaries — synthesise them into a unified narrative."
    )
    user_prompt = HumanMessage(
        f"Topic: {state['topic']}\n\nResearch summaries:\n{summaries}"
    )
    response = llm_synth.invoke([system, user_prompt])
    return {"final_report": response.content}

# ═══════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════
builder = StateGraph(ResearchState)

# Add the orchestrator nodes:
builder.add_node("planner",           planner_node)
builder.add_node("research_worker",   research_worker)   # will be called N times via Send
builder.add_node("synthesise",        synthesise_node)

# Linear flow: START → planner → fan_out → workers (parallel) → synthesise → END
builder.add_edge(START, "planner")

# fan_out is expressed as a conditional edge from planner that returns Send objects:
builder.add_conditional_edges(
    "planner",
    fan_out_node,    # returns list[Send] — LangGraph fans out automatically
    ["research_worker"],   # valid destination nodes
)

builder.add_edge("research_worker", "synthesise")
builder.add_edge("synthesise", END)

graph = builder.compile(checkpointer=MemorySaver())

# ═══════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════
config = {"configurable": {"thread_id": "research-001"}}

result = graph.invoke(
    {
        "topic":        "The current state of open-source large language models in 2025",
        "sub_queries":  [],
        "results":      [],
        "final_report": "",
    },
    config=config,
)

print(f"Sub-queries researched: {len(result['sub_queries'])}")
print(f"Results collected: {len(result['results'])}")
print("=== FINAL REPORT ===")
print(result["final_report"])
```

---

## How Results are Collected

When each `research_worker` invocation returns `{"results": [one_result]}`,
the `operator.add` reducer on `ResearchState.results` appends all of them:

```
Worker A returns: {"results": [{"query": "Q1", "summary": "..."}]}
Worker B returns: {"results": [{"query": "Q2", "summary": "..."}]}
Worker C returns: {"results": [{"query": "Q3", "summary": "..."}]}

After all workers complete:
state["results"] = [
    {"query": "Q1", "summary": "..."},
    {"query": "Q2", "summary": "..."},
    {"query": "Q3", "summary": "..."},
]
```

The `synthesise_node` runs only after ALL workers complete — LangGraph waits.

---

## Comparing Static vs Dynamic Parallelism

```python
# Static (LCEL RunnableParallel) — fixed at design time:
from langchain_core.runnables import RunnableParallel
chain = RunnableParallel({
    "web":   search_web | parse,
    "docs":  search_docs | parse,
    "news":  search_news | parse,
})
# Only ever runs exactly 3 tasks.

# Dynamic (LangGraph Send) — determined at runtime:
def fan_out(state):
    return [Send("worker", {"q": q}) for q in state["queries"]]
# Runs N tasks where N is determined by the planner at runtime.
```

---

## Common Pitfalls

| Pitfall                                    | Symptom                                           | Fix                                                                                        |
| ------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Using `operator.add` on a dict field       | Results stored as flat key overwriting each other | Use `Annotated[list[dict], operator.add]` — collect in a list                              |
| No reducer on the results field            | Only the last worker's result is kept             | Always annotate with `Annotated[list, operator.add]`                                       |
| Too many parallel workers (100+)           | Rate limit errors, memory pressure                | Add a concurrency limit or batch into groups of 10-20                                      |
| Worker state missing required fields       | `KeyError` in worker node                         | Always pass all required WorkerState fields in each `Send()`                               |
| Fan-out node runs when no items to process | Empty Send list = graph exits immediately         | Add a guard: `if not state["sub_queries"]: return {"final_report": "nothing to research"}` |

---

## Mini Summary

- `Send("node", state_dict)` creates one independent concurrent invocation of the named node
- A fan-out function returns `list[Send]` — LangGraph runs all of them concurrently
- Results are collected via the `operator.add` reducer — every worker appends to the shared list
- The synthesise node runs only after ALL parallel workers complete
- Use `Send` when the number of parallel tasks is unknown at graph design time

---

[← Human-in-the-Loop](02-human-in-the-loop.md) | [Next → Retry and Fallback](04-retry-and-fallback.md)
