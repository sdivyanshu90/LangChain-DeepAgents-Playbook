# Module 3.5 — DeepAgents Architecture

> **Why this module exists:** A ReAct agent is Level 1 autonomy. Production systems —
> those that conduct multi-day research, manage complex workflows, delegate to specialists,
> and recover from failure without human intervention — operate at Level 5.
> This module maps the journey from Level 1 to Level 5 and builds the patterns that define
> each level.

---

## Topics

| #   | File                                                         | What you will learn                                                                        |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 01  | [What Makes an Agent "Deep"](01-what-makes-an-agent-deep.md) | The 5-level progression; comparing shallow vs deep on the same task                        |
| 02  | [Supervisor Pattern](02-supervisor-pattern.md)               | Supervisor node, TaskPlan, routing to sub-agents via `Command`                             |
| 03  | [Swarm Handoff Pattern](03-swarm-handoff-pattern.md)         | Peer-to-peer handoff with `transfer_to_X()` tools; when to use vs Supervisor               |
| 04  | [Plan and Execute](04-plan-and-execute.md)                   | Planner produces `TaskPlan`; Executor processes steps; updating the plan as results arrive |
| 05  | [Reflexion Pattern](05-reflexion-pattern.md)                 | Generate → critique → revise loop; Reviewer node; revision history in state                |

---

## Level 5 DeepAgent Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        LEVEL 5 DEEPAGENT                                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐            │
│  │                        PLANNER                                  │            │
│  │  Reads goal + long-term memory → creates TaskPlan               │            │
│  └──────────────────────────┬────────────────────────────────────┘            │
│                             │ TaskPlan                                          │
│                             ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐            │
│  │                       SUPERVISOR                                │            │
│  │  Receives goal → creates sub-tasks → routes to sub-agents       │            │
│  │  via Command(goto="agent_name")                                 │            │
│  └──────┬──────────────────┬──────────────────┬───────────────────┘            │
│         │                  │                  │                                 │
│         ▼                  ▼                  ▼                                 │
│   ┌───────────┐      ┌───────────┐      ┌───────────┐                          │
│   │ RESEARCHER│      │  ANALYST  │      │  WRITER   │  ← specialist sub-agents │
│   └───────────┘      └───────────┘      └───────────┘                          │
│         │                  │                  │                                 │
│         └──────────────────┴──────────────────┘                                 │
│                             │ results                                            │
│                             ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐            │
│  │                       REFLECTOR                                 │            │
│  │  Scores output quality → routes to WRITER if revision needed    │            │
│  └──────────────────────────┬────────────────────────────────────┘            │
│                             │ if score >= threshold                             │
│                             ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐            │
│  │                    HUMAN GATE (optional)                        │            │
│  │  interrupt() → human approves → final delivery                  │            │
│  └─────────────────────────────────────────────────────────────────┘            │
│                                                                                  │
│  Long-term memory (vector store) ←→ injected into Planner and sub-agents       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Packages

```
langgraph>=0.3
langchain-core>=0.3
langchain-openai>=0.2
pydantic>=2.0
```
