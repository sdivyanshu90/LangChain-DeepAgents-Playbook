[← Retry and Fallback](04-retry-and-fallback.md) | [Next Module → DeepAgents Architecture](../../module-3.5-deepagents-architecture/README.md)

---

# 05 — Termination Conditions

## Why Termination Must Be Explicit

An agent that can loop must also be able to stop. Without an explicit termination
strategy, a pathological sequence of events produces an infinite loop:

1. Model produces a tool call
2. Tool returns an error
3. Model tries a different tool call
4. That tool also errors (or is unavailable)
5. Model loops back to step 1 — forever

In production, infinite loops exhaust LLM quota, produce massive LangSmith traces,
and block the thread until an operator kills the process. Explicit termination
converts a potential disaster into a graceful degradation.

---

## Real-World Analogy

A ship navigation system has a "maximum deviation allowed" parameter. If the autopilot
detects it has been correcting course 47 times without reaching the waypoint, it
sounds an alarm and hands control to the human crew rather than continuing to correct
indefinitely.

`MAX_ITERATIONS` is that parameter. The routing function is the autopilot's alarm logic.

---

## The Four Termination Conditions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TERMINATION CONDITIONS                                                     │
│                                                                             │
│  1. Natural completion                                                      │
│     Model produces a final answer (no tool_calls in response)              │
│     → routing function returns END                                          │
│                                                                             │
│  2. MAX_ITERATIONS guard                                                    │
│     iteration_count >= MAX_ITERATIONS                                       │
│     → routing function returns END with a partial/timeout response         │
│                                                                             │
│  3. finish_reason field in State                                            │
│     Node explicitly sets state["finish_reason"] = "success"|"timeout"|     │
│     "error"|"human_terminated"                                              │
│     → routing function routes on this field                                │
│                                                                             │
│  4. No-progress detection                                                   │
│     Same tool called with same args N times in a row                       │
│     → routing function detects stuck loop and terminates                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pattern 1 — MAX_ITERATIONS Guard

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

MAX_ITERATIONS = 10

class AgentState(TypedDict):
    messages:        Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    finish_reason:   Optional[str]   # "natural" | "max_iterations" | "error"

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]

    # Condition 1: MAX_ITERATIONS
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "timeout_node"

    # Condition 2: Natural completion (no tool calls)
    if not getattr(last, "tool_calls", None):
        return END

    # Continue looping
    return "tools"
```

---

## Pattern 2 — `finish_reason` Field

Using an explicit `finish_reason` field makes the termination state inspectable
after the run and filterable in LangSmith:

```python
def agent_node(state: AgentState) -> dict:
    """Agent node with explicit finish_reason tracking."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        # Don't call the LLM again — just set finish_reason and exit
        return {
            "finish_reason": "max_iterations",
            "messages": [AIMessage(
                content="I was unable to complete this task within the allowed steps. "
                        "Here is what I found so far: [partial results]"
            )],
        }

    response = llm_with_tools.invoke(state["messages"])
    new_count = state["iteration_count"] + 1

    if not response.tool_calls:
        finish = "natural"
    else:
        finish = "in_progress"

    return {
        "messages":        [response],
        "iteration_count": new_count,
        "finish_reason":   finish,
    }

def routing_fn(state: AgentState) -> str:
    reason = state.get("finish_reason", "in_progress")
    if reason == "natural":
        return END
    if reason == "max_iterations":
        return END
    if reason == "error":
        return "error_handler"
    return "tools"   # "in_progress"
```

---

## Pattern 3 — No-Progress Detection

```python
from collections import Counter

def detect_stuck_loop(state: AgentState) -> bool:
    """
    Detect if the agent is stuck calling the same tool with the same arguments
    repeatedly without making progress.
    Returns True if stuck.
    """
    # Examine the last N messages for repeated tool calls
    recent_messages = state["messages"][-10:]
    tool_call_signatures = []
    for msg in recent_messages:
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                # Create a hashable signature from name + sorted args
                sig = (tc["name"], str(sorted(tc["args"].items())))
                tool_call_signatures.append(sig)

    if not tool_call_signatures:
        return False

    # If any signature repeats 3+ times, the agent is stuck
    counts = Counter(tool_call_signatures)
    return any(count >= 3 for count in counts.values())

def enhanced_routing(state: AgentState) -> str:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "timeout_node"

    if detect_stuck_loop(state):
        return "stuck_loop_handler"

    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END

    return "tools"
```

---

## Graceful Termination Nodes

When an agent terminates due to exhausted iterations or a stuck loop, you want a
graceful summary, not an empty response:

```python
def timeout_node(state: AgentState) -> dict:
    """
    Called when MAX_ITERATIONS is reached.
    Produces a partial results summary rather than failing silently.
    """
    # Count what was accomplished before timeout
    tool_results = [
        m for m in state["messages"]
        if hasattr(m, "tool_call_id")  # ToolMessages
    ]
    completed_steps = len(tool_results)

    timeout_message = AIMessage(
        content=(
            f"I was unable to fully complete this task within {MAX_ITERATIONS} steps "
            f"({completed_steps} steps were completed). "
            "Here is a summary of what was accomplished:\n\n"
            + _summarise_partial_results(state)
        )
    )
    return {
        "messages":      [timeout_message],
        "finish_reason": "max_iterations",
    }

def stuck_loop_handler(state: AgentState) -> dict:
    """Called when the same tool call repeats 3+ times without progress."""
    stuck_message = AIMessage(
        content=(
            "I appear to be stuck in a loop attempting the same action repeatedly. "
            "This may indicate that the requested information is unavailable or that "
            "my approach needs to change. I'm stopping here to avoid wasting resources. "
            "Please try rephrasing your request or providing additional context."
        )
    )
    return {
        "messages":      [stuck_message],
        "finish_reason": "stuck_loop",
    }

def _summarise_partial_results(state: AgentState) -> str:
    """Extract partial results from completed tool calls."""
    from langchain_core.messages import ToolMessage
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if not tool_messages:
        return "No results were obtained before timeout."
    summaries = [f"- {tm.content[:200]}" for tm in tool_messages[:5]]
    return "Partial results obtained:\n" + "\n".join(summaries)
```

---

## Logging Loop Counts with LangSmith

```python
from langchain_core.runnables import RunnableConfig
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-agent-project"

def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    new_count = state["iteration_count"] + 1

    # Log the iteration count as run metadata (visible in LangSmith):
    # In a LangGraph context, this is done via run metadata in the config
    return {
        "messages":        [response],
        "iteration_count": new_count,
    }

# Attach metadata at invoke time:
config = {
    "configurable": {"thread_id": "session-001"},
    "metadata": {
        "session_id":  "session-001",
        "user_id":     "user-123",
        "max_allowed": MAX_ITERATIONS,
    },
    "tags": ["agent", "react", "production"],
    "run_name": "ReAct Agent — Session 001",
}

result = graph.invoke(initial_state, config=config)
```

---

## Complete State with All Termination Fields

```python
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class FullyTerminatedState(TypedDict):
    # Core agent state
    messages:        Annotated[list[BaseMessage], add_messages]

    # Loop control
    iteration_count: int
    max_iterations:  int   # set in initial state; never modified

    # Termination signal
    finish_reason: Optional[str]
    # Valid values:
    #   "natural"         — model answered without tool calls
    #   "max_iterations"  — hit iteration limit
    #   "stuck_loop"      — same tool call repeated 3+ times
    #   "tool_error"      — irrecoverable tool failure
    #   "human_cancelled" — human terminated via interrupt
    #   None              — still running

    # Output
    final_answer: Optional[str]
```

---

## Common Pitfalls

| Pitfall                                          | Symptom                                          | Fix                                                       |
| ------------------------------------------------ | ------------------------------------------------ | --------------------------------------------------------- |
| No `MAX_ITERATIONS` guard                        | Agent loops until quota exhaustion or OOM        | Always add `iteration_count >= MAX_ITERATIONS` check      |
| MAX_ITERATIONS too small                         | Legitimate multi-step tasks time out             | Profile typical tasks; set to 2× the expected steps       |
| Timeout node raises instead of returning message | Process crashes at the worst moment              | Timeout node must always return a valid state update      |
| No `finish_reason` field                         | Hard to distinguish success from timeout in logs | Always set `finish_reason`; filter on it in LangSmith     |
| Stuck-loop detection too sensitive               | Single retry incorrectly classified as stuck     | Require ≥3 repetitions of the exact same call, not just 2 |

---

## Mini Summary

- Every agent loop needs an explicit termination strategy — never rely on the model to stop itself
- `MAX_ITERATIONS` is the baseline safety valve: route to a timeout node when reached
- `finish_reason` in State documents _why_ the agent stopped — invaluable in production debugging
- No-progress detection (`detect_stuck_loop`) catches infinite loops that MAX_ITERATIONS alone won't prevent
- Termination nodes must produce a graceful summary response — silence or exceptions are not acceptable outcomes

---

[← Retry and Fallback](04-retry-and-fallback.md) | [Next Module → DeepAgents Architecture](../../module-3.5-deepagents-architecture/README.md)
