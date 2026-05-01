[← LangSmith Setup](01-langsmith-setup.md) | [Next → Trace Inspection](03-trace-inspection-and-replay.md)

---

# 02 — Run Metadata and Tags

## Why Metadata Matters

A LangSmith project with 500 runs and no metadata is almost impossible to navigate.
You can't filter by user, session, model version, or feature. When a bug report arrives
("the agent gave wrong output this morning"), you can't find the specific run.

Metadata and tags turn the trace list into a searchable, filterable database.

---

## Real-World Analogy

A hospital's patient records. Without metadata — patient ID, attending doctor,
procedure date — the records are unsearchable. With metadata, a doctor can instantly
pull "all patients seen by Dr. Kim in June with a knee injury." Metadata turns a pile
of documents into a system.

---

## Tags vs Metadata — When to Use Each

```
Tags: string labels for categorical filtering
      ─────────────────────────────────────────
      Examples: ["production", "v2", "module-3.6", "gpt-4o-mini"]
      LangSmith UI: filter by tag
      Best for: environment, model, version, feature, team

Metadata: key-value pairs for structured data
          ──────────────────────────────────────
          Examples: {"user_id": "u123", "session_id": "s456", "iteration": 3}
          LangSmith UI: filter by metadata key/value
          Best for: IDs, counts, scores, anything you'd query programmatically
```

---

## Adding Metadata at Invoke Time

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

# Metadata and tags are passed via the config dict
result = llm.invoke(
    [HumanMessage("Explain embeddings.")],
    config={
        "run_name": "explain-embeddings-v2",
        "tags": ["production", "education", "gpt-4o-mini"],
        "metadata": {
            "user_id":        "user-789",
            "session_id":     "session-2025-06",
            "prompt_version": "v2",
            "feature_flag":   "new-formatting",
        },
    },
)
```

---

## RunnableConfig for Reusable Configuration

```python
from langchain_core.runnables import RunnableConfig
import uuid

def make_run_config(
    user_id: str,
    session_id: str,
    task_name: str,
    tags: list[str] | None = None,
) -> RunnableConfig:
    """
    Factory for consistent run configuration across the application.
    Centralises run_name and metadata conventions.
    """
    return RunnableConfig(
        run_name=f"{task_name}:{session_id}",
        tags=(tags or []) + ["production", "v3"],
        metadata={
            "user_id":    user_id,
            "session_id": session_id,
            "task_name":  task_name,
            "run_id":     str(uuid.uuid4()),
        },
    )

# Usage:
config = make_run_config(
    user_id="user-123",
    session_id="session-456",
    task_name="research-agent",
    tags=["research", "gpt-4o"],
)
result = graph.invoke(initial_state, config=config)
```

---

## Attaching Metadata to Individual LangGraph Nodes

Sometimes you want metadata at the graph level, but also at individual node level.
For example: log the `iteration_count` as metadata on each executor node call.

```python
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler

class IterationMetadataCallback(BaseCallbackHandler):
    """Adds dynamic metadata to each LLM call based on current state."""

    def __init__(self, iteration: int, step_description: str):
        self.iteration = iteration
        self.step_description = step_description

    def on_llm_start(self, serialized, prompts, **kwargs):
        # This metadata appears on the LLM child span in LangSmith
        if "invocation_params" in kwargs:
            kwargs["invocation_params"]["metadata"] = {
                "iteration":        self.iteration,
                "step_description": self.step_description,
            }

def executor_node(state, config: RunnableConfig) -> dict:
    """
    Pass per-node metadata via callbacks in the config.
    The config parameter is automatically injected by LangGraph
    when the node function accepts it.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create a node-specific config with iteration metadata
    node_config = {
        **config,
        "callbacks": [
            IterationMetadataCallback(
                iteration=state["iteration"],
                step_description=f"Executing step {state['iteration']}",
            )
        ],
        "metadata": {
            **config.get("metadata", {}),
            "node":      "executor",
            "iteration": state["iteration"],
        },
    }

    response = llm.invoke(state["messages"], config=node_config)
    return {"messages": [response]}
```

**Note:** LangGraph automatically injects `config: RunnableConfig` into node functions
when declared as a parameter. This is the idiomatic way to access and extend config
inside nodes.

---

## Propagating Session ID Through a Multi-Turn Graph

In a production agent, a single user session spans many graph invocations. The
`thread_id` in MemorySaver is the session identifier — always use it as your primary
session identifier:

```python
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())

def run_agent_turn(user_message: str, user_id: str, thread_id: str) -> str:
    """
    Each turn in a multi-turn conversation.
    All turns share the same thread_id AND the same LangSmith session_id.
    """
    config = RunnableConfig(
        run_name=f"agent-turn:{thread_id}",
        tags=["multi-turn", "production"],
        metadata={
            "user_id":   user_id,
            "thread_id": thread_id,  # LangSmith: filter all turns by thread
        },
        configurable={"thread_id": thread_id},  # LangGraph MemorySaver key
    )
    from langchain_core.messages import HumanMessage
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content

# Two turns — same thread_id, same LangSmith metadata.thread_id
turn1 = run_agent_turn("What is RAG?",     "user-001", "thread-abc")
turn2 = run_agent_turn("Give me an example", "user-001", "thread-abc")
```

In LangSmith: filter by `metadata.thread_id:thread-abc` to see both turns together.

---

## Metadata Conventions for Production

```python
# Recommended metadata schema for production LangGraph agents:
STANDARD_METADATA = {
    # Identity
    "user_id":        "uuid",      # who initiated the request
    "session_id":     "uuid",      # browser/app session
    "thread_id":      "uuid",      # LangGraph memory thread

    # Versioning
    "agent_version":  "v3.2",      # your agent code version
    "model":          "gpt-4o",    # model used
    "prompt_version": "v5",        # prompt template version

    # Context
    "environment":    "production", # "production" | "staging" | "test"
    "feature":        "research-agent",
    "iteration":      0,            # for multi-step nodes

    # Cost control
    "max_tokens_hint": 2000,        # hint (not enforced); for dashboards
}
```

---

## Filtering in the LangSmith UI

Once metadata is set, LangSmith supports powerful filtering:

```
# UI search bar queries:
user_id=user-123
metadata.session_id=session-456
tag=production
status=error
run_name contains "research"
latency > 5000        # ms
cost > 0.01           # dollars

# Combine:
tag=production AND metadata.user_id=user-123 AND status=error
```

---

## Common Pitfalls

| Pitfall                                 | Symptom                                 | Fix                                                                     |
| --------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------- |
| No `session_id` in metadata             | Can't group related runs                | Always include `session_id` and `thread_id` in metadata                 |
| Tags with spaces or slashes             | UI filtering fails or shows errors      | Use lowercase hyphen-separated tags: `"my-tag"` not `"My Tag"`          |
| Metadata values are not strings         | Filtering fails for non-string values   | Convert all metadata values to strings or numbers; no nested dicts      |
| Config not forwarded from graph to node | Node metadata is missing in child spans | Accept `config: RunnableConfig` parameter in node; merge and forward it |
| Different `run_name` format each call   | Hard to correlate runs                  | Use a consistent naming convention: `"{task}:{session_id}"`             |

---

## Mini Summary

- Tags are categorical labels for filtering; metadata is structured key-value data for querying
- Use `RunnableConfig` to create reusable, consistent run configuration across the application
- Nodes can receive and extend `config: RunnableConfig` to add per-node metadata
- Always include `user_id`, `session_id`, and `thread_id` in production metadata
- Use lowercase hyphen-separated tags; convert all metadata values to strings or numbers

---

[← LangSmith Setup](01-langsmith-setup.md) | [Next → Trace Inspection](03-trace-inspection-and-replay.md)
