# Level 3: Tool Calling, Workflow Design, and DeepAgents

Level 3 is where the repository stops teaching isolated components and starts teaching autonomous systems.

The focus now is not just how to call a model, retrieve context, or hold memory. The focus is how to design durable workflows that can plan, call tools, route work, recover from failure, and expose what they are doing while they do it.

## Why This Level Matters

Simple chains and RAG applications solve a narrow class of problems well. But many real tasks require more:

- multiple tool calls
- branching decisions
- long-running task state
- retries and recovery
- delegation between specialized roles
- checkpoints and observability

This level teaches the engineering patterns behind those systems.

## Modules

- [Module 3.1: Tool calling fundamentals](modules/module-3.1-tool-calling-fundamentals/README.md)
- [Module 3.2: Custom tools](modules/module-3.2-custom-tools/README.md)
- [Module 3.3: LangGraph basics](modules/module-3.3-langgraph-basics/README.md)
- [Module 3.4: Multi-step agent workflows](modules/module-3.4-multi-step-agent-workflows/README.md)
- [Module 3.5: DeepAgents architecture](modules/module-3.5-deepagents-architecture/README.md)
- [Module 3.6: Observability and debugging](modules/module-3.6-observability-and-debugging/README.md)

## Projects

- [Project 3.1: Autonomous Research Assistant](projects/project-3.1-autonomous-research-assistant/README.md)
- [Project 3.2: Multi-tool Travel Planner](projects/project-3.2-multi-tool-travel-planner/README.md)
- [Project 3.3: Incident Triage Agent](projects/project-3.3-incident-triage-agent/README.md)
- [Project 3.4: Customer Support Triage Agent](projects/project-3.4-customer-support-triage-agent/README.md)
- [Project 3.5: Codebase Explorer](projects/project-3.5-codebase-explorer/README.md)
- [Project 3.6: Sales Intelligence Agent](projects/project-3.6-sales-intelligence-agent/README.md)
- [Project 3.7: Meeting-to-Action Agent](projects/project-3.7-meeting-to-action-agent/README.md)
- [Project 3.8: Data Query Agent](projects/project-3.8-data-query-agent/README.md)
- [Project 3.9: Compliance Review Assistant](projects/project-3.9-compliance-review-assistant/README.md)
- [Project 3.10: DeepAgents Orchestrator](projects/project-3.10-deepagents-orchestrator/README.md)
- [Project 3.11: Autonomous Content Ops Agent](projects/project-3.11-autonomous-content-ops-agent/README.md)
- [Project 3.12: Workflow Recovery Agent](projects/project-3.12-workflow-recovery-agent/README.md)

## Recommended Order

1. Start with Modules 3.1 and 3.2 to understand when and how tools should exist.
2. Read Module 3.3 before building any serious autonomous workflow.
3. Continue to Module 3.4 to learn retries, branching, and local recovery.
4. Study Module 3.5 to understand what makes an agent deep rather than shallow.
5. Read Module 3.6 before trusting any autonomous system in production.
6. Build the projects in order if you want the smoothest progression from simple routing to orchestration.

## Shared Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r deepagents/requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## Learning Outcome

By the end of Level 3, you should be able to explain and implement:

- model-directed tool use and typed tools
- LangGraph state, nodes, edges, and conditional routing
- multi-step workflows with retries and reflection
- autonomous loops with guardrails and checkpoints
- specialized sub-agent coordination patterns
- debugging and tracing strategies for agentic systems

This level is the bridge from assistant-style apps to engineered autonomous workflows.
