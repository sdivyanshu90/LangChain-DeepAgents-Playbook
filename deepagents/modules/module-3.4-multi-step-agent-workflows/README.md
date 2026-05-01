# Module 3.4 — Multi-Step Agent Workflows

> **Why this module exists:** A single-node agent can answer questions. But real-world
> tasks — research reports, content pipelines, incident resolution — require many sequential
> steps with feedback loops, human checkpoints, parallel sub-tasks, and graceful failure recovery.
> This module shows you how to build agents that work at that scale.

---

## Topics

| #   | File                                                     | What you will learn                                                          |
| --- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 01  | [ReAct Loop From Scratch](01-react-loop-from-scratch.md) | Build a ReAct agent in LangGraph without shortcuts; every decision explained |
| 02  | [Human-in-the-Loop](02-human-in-the-loop.md)             | `interrupt()`, NodeInterrupt, checkpointing, resuming with `Command`         |
| 03  | [Send API Fan-Out](03-send-api-fan-out.md)               | Dynamic parallelism with `Send`; research fan-out to N sub-queries           |
| 04  | [Retry and Fallback](04-retry-and-fallback.md)           | Retry counter in State, routing on failure, fallback nodes                   |
| 05  | [Termination Conditions](05-termination-conditions.md)   | MAX_ITERATIONS guard, finish_reason, preventing infinite loops               |

---

## Module Architecture — The ReAct Loop

```
        START
          │
          ▼
    ┌──────────┐
    │  REASON  │  ← LLM reasons over messages + tool results
    │  (agent) │
    └────┬─────┘
         │
    tool_calls?
    ┌────┴────┐
   YES       NO
    │         │
    ▼         ▼
┌──────────┐  END
│  ACT     │  ← execute tool calls
│  (tools) │
└────┬─────┘
     │
  OBSERVE
  (ToolMessages appended to state)
     │
     └──────▶ back to REASON

The loop terminates when:
  1. The model produces no tool_calls  →  final answer reached
  2. retry_count > MAX_RETRIES         →  fallback response
  3. iteration_count > MAX_ITERATIONS  →  safety termination
```

---

## Key Packages

```
langgraph>=0.3
langchain-core>=0.3
langchain-openai>=0.2
```
