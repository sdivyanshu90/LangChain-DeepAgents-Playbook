# Module 3.3 — LangGraph Core

> **Why this module exists:** LCEL chains are powerful for linear, deterministic pipelines.
> But when you need state that persists across steps, conditional branching, cycles,
> interrupts for human review, or fault-tolerant resumption — LCEL breaks down.
> LangGraph was built specifically for stateful, multi-step, cyclical workflows.

---

## Topics

| #   | File                                                         | What you will learn                                                                    |
| --- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| 01  | [StateGraph Anatomy](01-stategraph-anatomy.md)               | TypedDict State, Annotated reducers, `add_messages`, compiled graph                    |
| 02  | [Nodes as Pure Functions](02-nodes-as-pure-functions.md)     | Node signature, partial state returns, update-merge semantics                          |
| 03  | [Edges and Routing](03-edges-and-routing.md)                 | `add_edge` vs `add_conditional_edges`, routing functions, `END`, Mermaid visualisation |
| 04  | [MemorySaver Checkpointing](04-memorysaver-checkpointing.md) | Persistence, `thread_id`, `interrupt()`, resuming with `Command`                       |
| 05  | [Building a Complete Graph](05-building-a-complete-graph.md) | Full 3-node triage graph: every line annotated                                         |

---

## Module Architecture — StateGraph Anatomy

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STATEGRAPH ANATOMY                                   │
│                                                                              │
│   State (TypedDict)                                                          │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │  messages: Annotated[list[BaseMessage], add_messages]│                  │
│   │  topic:    str                                       │                  │
│   │  retries:  Annotated[int, lambda a, b: b]            │                  │
│   └─────────────────────────────────────────────────────┘                   │
│            │                                                                 │
│            ▼ passed to every node                                            │
│                                                                              │
│  START ──▶ [node_A] ──▶ [node_B] ──▶ [node_C] ──▶ END                      │
│                │                         ▲                                   │
│                │    (conditional edge)   │                                   │
│                └─── should_retry? ───────┘                                   │
│                          │                                                   │
│                          ▼ no                                                │
│                        END                                                   │
│                                                                              │
│  MemorySaver  ─────────────────────────────────────────────────────────────  │
│  After every node execution, the full State is checkpointed to memory.       │
│  thread_id isolates state per session. interrupt() pauses here for human     │
│  review. Command(resume=value) resumes from the checkpoint.                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph vs LCEL Comparison

| Capability                   | LCEL    | LangGraph |
| ---------------------------- | ------- | --------- |
| Linear pipeline              | ✅      | ✅        |
| Shared mutable state         | ❌      | ✅        |
| Cycles / loops               | ❌      | ✅        |
| Conditional branching        | Limited | ✅        |
| Persist mid-run              | ❌      | ✅        |
| Human-in-the-loop interrupts | ❌      | ✅        |
| Parallel fan-out (Send API)  | Limited | ✅        |
| Fault-tolerant resumption    | ❌      | ✅        |

## Key Packages

```
langgraph>=0.3
langchain-core>=0.3
langchain-openai>=0.2
```
