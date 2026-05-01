[← Supervisor Pattern](02-supervisor-pattern.md) | [Next → Plan and Execute](04-plan-and-execute.md)

---

# 03 — Swarm Handoff Pattern

## Why Peer-to-Peer Instead of Supervisor?

The Supervisor pattern has a bottleneck: every action requires the Supervisor to route
it. For highly dynamic conversations — where the right specialist depends on content
rather than task type, and transitions happen mid-conversation — the Supervisor adds
latency and complexity for little gain.

The Swarm pattern removes the central authority. Each agent has `transfer_to_X()`
tools that produce a `Command(goto="X")`, passing the full conversation context at
handoff. The agent that should handle the next message decides by calling a transfer
tool — no external routing logic needed.

---

## Real-World Analogy

A doctor's practice where specialists transfer patients directly. A GP sees a patient,
determines they need a dermatologist, and personally walks them over to Dr. Kim with
a verbal briefing ("patient has had this rash for 2 weeks, no allergies, here's the
context"). Dr. Kim handles the case; if she needs a biopsy, she walks the patient to
the pathology lab herself. There is no central receptionist routing every handoff.

The `transfer_to_X()` tool is the "walk the patient over personally" action.

---

## Core Architecture

```
┌─────────────┐   transfer_to_analyst()   ┌──────────────┐
│  RESEARCHER │ ─────────────────────────► │   ANALYST    │
│             │◄─────────────────────────  │              │
└─────────────┘  transfer_to_researcher() └──────┬───────┘
       ▲                                          │ transfer_to_writer()
       │                                          ▼
       │                                   ┌──────────────┐
       └────────────────────────────────── │    WRITER    │
              transfer_to_researcher()     └──────────────┘

No central supervisor. Each agent has tools to hand off to any other agent.
Full conversation history is passed at every handoff.
```

---

## The Handoff Tool Pattern

Each agent's handoff tools return a `Command` that updates the graph state AND
routes to the target agent:

```python
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from typing import Annotated

# Handoff tools for the Researcher agent
@tool("transfer_to_analyst")
def transfer_to_analyst(
    summary: Annotated[str, "Summary of research findings to pass to the analyst"]
) -> Command:
    """
    Transfer control to the analyst agent when research is complete.
    Use this when you have gathered sufficient data and need it interpreted.
    """
    return Command(
        goto="analyst",
        update={
            "messages": [
                ToolMessage(
                    content=f"[HANDOFF → ANALYST]: {summary}",
                    tool_call_id="handoff",  # required field
                )
            ],
            "active_agent": "analyst",
        },
    )

@tool("transfer_to_writer")
def transfer_to_writer(
    brief: Annotated[str, "Creative brief and all findings to pass to the writer"]
) -> Command:
    """
    Transfer control to the writer agent when analysis is complete.
    """
    return Command(
        goto="writer",
        update={
            "messages": [
                ToolMessage(
                    content=f"[HANDOFF → WRITER]: {brief}",
                    tool_call_id="handoff",
                )
            ],
            "active_agent": "writer",
        },
    )
```

---

## Full State Design

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SwarmState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    active_agent: str   # which agent is currently in control
    goal:         str
    handoff_count: int  # prevents infinite handoff loops
```

---

## Building Each Agent Node

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Researcher Node ────────────────────────────────────────────────────────────
def researcher_node(state: SwarmState) -> dict:
    """
    Researcher has: web_search + transfer_to_analyst tools.
    Works until it decides to hand off.
    """
    from langchain_core.tools import tool

    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""
        return f"[MOCK SEARCH RESULT for '{query}']"

    researcher_tools = [web_search, transfer_to_analyst, transfer_to_writer]
    researcher_llm = llm.bind_tools(researcher_tools)

    system = SystemMessage(
        "You are a research specialist. Gather information for the goal.\n"
        "When you have enough data, call transfer_to_analyst with a summary."
    )
    response = researcher_llm.invoke([system, *state["messages"]])
    return {"messages": [response]}

# ── Analyst Node ───────────────────────────────────────────────────────────────
def analyst_node(state: SwarmState) -> dict:
    """Analyst interprets research. Can hand off to writer or back to researcher."""
    from langchain_core.tools import tool

    @tool("transfer_to_researcher")
    def transfer_to_researcher_tool(
        request: Annotated[str, "What additional research is needed"]
    ) -> Command:
        """Transfer back to researcher if more data is needed."""
        return Command(
            goto="researcher",
            update={
                "messages": [ToolMessage(content=f"[HANDOFF → RESEARCHER]: {request}", tool_call_id="handoff")],
                "active_agent": "researcher",
            },
        )

    analyst_tools = [transfer_to_researcher_tool, transfer_to_writer]
    analyst_llm = llm.bind_tools(analyst_tools)

    system = SystemMessage(
        "You are an analysis specialist. Interpret the research findings.\n"
        "When analysis is complete, call transfer_to_writer with a detailed brief.\n"
        "If you need more research, call transfer_to_researcher."
    )
    response = analyst_llm.invoke([system, *state["messages"]])
    return {"messages": [response]}

# ── Writer Node ────────────────────────────────────────────────────────────────
def writer_node(state: SwarmState) -> dict:
    """Writer produces the final deliverable. Does not hand off — ends the flow."""
    system = SystemMessage(
        "You are a professional writer. Produce the final deliverable "
        "based on all the research and analysis in the conversation. "
        "Do not call any tools — write the complete output directly."
    )
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}
```

---

## Routing in a Swarm Graph

Because agents use `Command(goto=...)`, LangGraph routes them without conditional edges.
But you still need a top-level router to handle the _initial_ dispatch and to protect
against infinite loops:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

MAX_HANDOFFS = 6

def route_to_active_agent(state: SwarmState) -> str:
    """Route to whoever is the currently active agent."""
    if state["handoff_count"] >= MAX_HANDOFFS:
        return "writer"  # force to writer after too many handoffs
    return state["active_agent"]

# Tool node to process handoff tool calls before routing:
from langgraph.prebuilt import ToolNode

builder = StateGraph(SwarmState)
builder.add_node("researcher", researcher_node)
builder.add_node("analyst",    analyst_node)
builder.add_node("writer",     writer_node)

# Start at the researcher
builder.add_edge(START, "researcher")

# Command(goto=...) from transfer tools handles routing automatically
builder.add_edge("writer", END)

graph = builder.compile(checkpointer=MemorySaver())
```

---

## Swarm vs Supervisor — Decision Guide

```
Use SUPERVISOR when:
  ✓ Task types are known at design time (research, analyse, write)
  ✓ Tasks are sequential (research → analyse → write, in order)
  ✓ You need full visibility of all routing decisions in one node
  ✓ Sub-agents should not communicate context to each other
  ✓ Simpler debugging is a priority

Use SWARM when:
  ✓ Handoffs depend on the content of the conversation
  ✓ Transitions are dynamic and unpredictable
  ✓ Each agent is best positioned to decide who handles the next step
  ✓ You want lower latency (no Supervisor round-trip per step)
  ✓ The conversation flows like a real-world team (fluid delegation)

Avoid SWARM when:
  ✗ You can't define clear handoff criteria for each agent
  ✗ You've already seen infinite handoff loops in testing
  ✗ Audit requirements demand a central routing log
```

---

## Pitfall: Infinite Handoff Loops

The most common Swarm failure: Agent A hands to Agent B, which hands back to Agent A,
which hands back to Agent B...

Prevention strategies:

```python
# Strategy 1: handoff_count in state
def researcher_node(state: SwarmState) -> dict:
    if state["handoff_count"] >= MAX_HANDOFFS:
        # Force an answer instead of handing off
        return {"messages": [AIMessage(content="[RESEARCHER] Max handoffs reached. Delivering partial findings.")]}
    # ...normal logic

# Strategy 2: Prevent back-handoff to the same agent
# In transfer_to_researcher tool:
@tool("transfer_to_researcher")
def transfer_to_researcher_tool(request: str) -> Command:
    """Only transfer back to researcher if not already at researcher."""
    # Add a check in the tool description that discourages re-handoff:
    # "Only use this if you are the analyst and genuinely need more raw data."
    return Command(goto="researcher", ...)

# Strategy 3: Track handoff chain in state
# Add handoff_history: list[str] = [] to SwarmState
# If handoff_history[-1] == next_agent, produce output instead of handing off
```

---

## Common Pitfalls

| Pitfall                                       | Symptom                                             | Fix                                                                          |
| --------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------- |
| No MAX_HANDOFFS guard                         | Infinite loop; quota exhaustion                     | Add `handoff_count` to state; check in every node                            |
| `tool_call_id` missing in ToolMessage         | LangChain raises `ValueError`                       | Always set `tool_call_id` in `ToolMessage` inside transfer tools             |
| Full message history grows without bound      | Context window exceeded mid-conversation            | Trim messages to last N in each node, or summarise on handoff                |
| All agents have all transfer tools            | Ambiguous routing; model picks random agent         | Each agent should only have tools to transfer to agents that make sense next |
| Handoff tool defined inside the node function | New tool object on every call; schema inconsistency | Define transfer tools at module level, not inside node functions             |

---

## Mini Summary

- Swarm replaces Supervisor with peer-to-peer handoffs via `transfer_to_X()` tools returning `Command(goto=...)`
- Full conversation context is passed at every handoff — no information loss between agents
- Each agent decides when to hand off based on the content of the conversation
- Always add a `handoff_count` guard to prevent infinite loops
- Use Swarm for dynamic, content-driven flows; use Supervisor for sequential, structured workflows

---

[← Supervisor Pattern](02-supervisor-pattern.md) | [Next → Plan and Execute](04-plan-and-execute.md)
