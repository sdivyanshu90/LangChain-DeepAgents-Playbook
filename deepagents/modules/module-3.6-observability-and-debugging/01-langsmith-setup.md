[← Module Overview](README.md) | [Next → Run Metadata and Tags](02-run-metadata-and-tags.md)

---

# 01 — LangSmith Setup

## The Problem: Debugging Blind

Without observability, every agent failure forces you into guesswork. You add `print()`
statements, re-run the agent, discover the print is in the wrong node, add more prints,
re-run again. For a multi-step agent with 10+ nodes, this can take hours.

LangSmith solves this by capturing every run automatically — inputs, outputs, latency,
token counts, errors — with zero code changes in your agent. Set three environment
variables and every future run is traced.

---

## Real-World Analogy

A flight data recorder (black box) on an aeroplane. The pilots do not have to remember
to activate it before takeoff. It runs continuously, recording every instrument reading.
When something goes wrong, investigators pull the box and replay every second of the
flight.

LangSmith is the black box for your agent.

---

## Step 1 — Get Your API Key

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up / log in
3. Navigate to **Settings → API Keys → Create API Key**
4. Copy the key (starts with `lsv2_pt_...`)

**Free tier limits** (as of 2025):

- 5,000 traces per month
- 14-day trace retention
- 1 project
- No team collaboration features
- Sufficient for all exercises in this playbook

---

## Step 2 — Set Environment Variables

```bash
# .env (project root — never commit this file)
LANGSMITH_API_KEY=lsv2_pt_your_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-deepagents-playbook
OPENAI_API_KEY=sk-your-openai-key
```

Load the `.env` file in Python:

```python
from dotenv import load_dotenv
load_dotenv()

# Verify setup
import os
assert os.environ.get("LANGSMITH_TRACING") == "true", "Tracing not enabled"
assert os.environ.get("LANGSMITH_API_KEY"),            "No LangSmith API key"
print(f"Project: {os.environ.get('LANGSMITH_PROJECT', 'default')}")
```

---

## Step 3 — Verify With a Simple Run

```python
# verify_langsmith.py
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

# This single call will create a trace in LangSmith
response = llm.invoke(
    [HumanMessage(content="Say 'LangSmith tracing confirmed' and nothing else.")],
    config={
        "run_name": "verify-langsmith-setup",
        "tags":     ["setup", "verify"],
        "metadata": {"source": "01-langsmith-setup.md"},
    },
)
print(response.content)
print("\nCheck your LangSmith project — you should see a new trace.")
```

After running this, navigate to your LangSmith project:
`smith.langchain.com → Projects → langchain-deepagents-playbook`

You should see a run named `verify-langsmith-setup` with full input/output.

---

## Enabling and Disabling Per Run

Sometimes you want to trace only specific runs (e.g., in production, not every
unit test):

```python
import os
from langchain_core.runnables import RunnableConfig

# Method 1: Environment variable (affects ALL subsequent runs in the process)
os.environ["LANGSMITH_TRACING"] = "false"
# ... run without tracing ...
os.environ["LANGSMITH_TRACING"] = "true"

# Method 2: Per-run config (preferred — no global state mutation)
config_with_tracing = RunnableConfig(
    run_name="important-run",
    tags=["production", "v2"],
    metadata={"session_id": "abc123"},
)

config_no_tracing = RunnableConfig(
    callbacks=[],  # explicitly empty — no LangSmith callback
)

# Production run (traced)
result = llm.invoke([HumanMessage("Hello")], config=config_with_tracing)

# Test run (not traced)
result = llm.invoke([HumanMessage("Hello")], config=config_no_tracing)
```

---

## `run_name` and `tags` Parameters

`run_name` and `tags` appear in the LangSmith UI and are used for filtering.
Set them for every significant run:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

# Named run — easy to find in LangSmith
result = llm.invoke(
    [HumanMessage("Summarise the key risks of LLM agents.")],
    config={
        "run_name": "risk-analysis-2025-06",  # descriptive name; shows in UI
        "tags": [
            "module-3.6",       # which module
            "risk-analysis",    # task type
            "gpt-4o-mini",      # model used
        ],
        "metadata": {
            "user_id":     "user-123",
            "session_id":  "session-456",
            "prompt_version": "v2",
        },
    },
)
```

**Filtering runs by tag in LangSmith UI:**

- Navigate to your project
- Use the search bar: `tag:risk-analysis`
- Or filter by metadata key: `metadata.user_id:user-123`

---

## LangSmith UI Overview

```
smith.langchain.com
└── Projects
    └── langchain-deepagents-playbook
        ├── Runs              ← list of all runs; filter by tag, date, status
        │   └── [run-name]    ← click to see full trace tree
        │       ├── Input     ← what was sent to the model
        │       ├── Output    ← what the model returned
        │       ├── Metadata  ← your custom metadata
        │       ├── Feedback  ← programmatic scores (Module 3.6/04)
        │       └── Child runs ← sub-calls (nodes in a LangGraph run)
        │
        ├── Datasets          ← saved input/output pairs for evaluation
        ├── Experiments       ← evaluate a prompt variant against a dataset
        └── Monitoring        ← cost/latency dashboards (Module 3.6/05)
```

---

## LangGraph + LangSmith Integration

LangGraph automatically creates a trace hierarchy that matches the graph structure.
No extra configuration is needed — just set `LANGSMITH_TRACING=true`:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SimpleState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def my_node(state: SimpleState) -> dict:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(SimpleState)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile(checkpointer=MemorySaver())

# This single invoke call creates a full trace tree in LangSmith:
# graph_run → my_node → ChatOpenAI call
config = {
    "configurable": {"thread_id": "demo-001"},
    "run_name": "simple-graph-demo",
    "tags": ["demo", "module-3.6"],
}
result = graph.invoke(
    {"messages": [HumanMessage("What is LangSmith?")]},
    config=config,
)
```

In LangSmith, you will see:

```
simple-graph-demo (LangGraph)
└── my_node
    └── ChatOpenAI (gpt-4o-mini)
        input: [HumanMessage("What is LangSmith?")]
        output: AIMessage("LangSmith is a platform...")
        tokens: prompt=12, completion=47
        latency: 0.8s
```

---

## Common Pitfalls

| Pitfall                     | Symptom                                       | Fix                                                                     |
| --------------------------- | --------------------------------------------- | ----------------------------------------------------------------------- |
| `LANGSMITH_TRACING` not set | No traces appear in LangSmith                 | Set `LANGSMITH_TRACING=true` in `.env`, not just `LANGCHAIN_TRACING_V2` |
| Wrong project name          | Traces go to "default" project                | Set `LANGSMITH_PROJECT` to match your intended project name             |
| API key in code             | Security risk; key exposed in version control | Always use `.env` + `load_dotenv()`; never hardcode                     |
| Free tier exhausted         | Traces stop appearing after 5,000/month       | Use `tags` to selectively trace only important runs                     |
| No `run_name` set           | All runs appear as random IDs in the UI       | Always set `run_name` for production and important test runs            |

---

## Mini Summary

- Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT` in `.env`
- Call `load_dotenv()` before any LangChain code — tracing starts automatically
- Use `run_name`, `tags`, and `metadata` in every significant run for easy filtering
- LangGraph creates the trace hierarchy automatically — no extra configuration needed
- The free tier (5,000 traces/month) is sufficient for all exercises in this playbook

---

[← Module Overview](README.md) | [Next → Run Metadata and Tags](02-run-metadata-and-tags.md)
