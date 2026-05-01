# LangChain: Zero to Hero — with DeepAgents

> **From your first LLM call to a fully autonomous, multi-actor cognitive system — step by step, with the *why* explained at every stage.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple)](https://langchain-ai.github.io/langgraph/)
[![LangSmith](https://img.shields.io/badge/LangSmith-tracing-orange)](https://smith.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Mission

Most AI tutorials stop too early. They show a prompt, a chain, or a quick demo, then leave out every architecture decision that matters when you need a system to be **reliable, testable, observable, and maintainable** in production.

This repository exists to close that gap.

It follows a **why-first philosophy**: no abstraction is introduced without first explaining the failure mode it was invented to solve. Every module opens with a real-world analogy, walks through internal mechanics, and then shows production-grade, fully commented code.

The goal is not to memorise APIs. The goal is to build **developer intuition** for how modern agentic AI systems behave, where they fail, and how to engineer them well.

---

## Who This Is For

| Audience | What you will get |
|---|---|
| LangChain beginners | A modern, structured path from zero — no legacy patterns |
| Intermediate builders | Production RAG, tool-calling, and stateful agent patterns |
| LangGraph practitioners | Multi-actor architectures, Supervisor/Swarm patterns, checkpointing |
| Teams and educators | Reusable project templates adapted for real products and workshops |

**Prerequisites:**
- Python 3.11+ installed
- Comfort with functions, classes, dicts, and virtual environments
- A general understanding of APIs and environment variables
- Access to at least one LLM provider API key (OpenAI recommended to start)
- Optional: LangSmith account for tracing in advanced modules

---

## The DeepAgents Progression — Level 0 to Level 5

Every module and project in this curriculum explicitly locates itself within this hierarchy.
Understanding which level a system operates at — and *why* moving up one level solves a problem the lower level cannot handle — is the core architectural skill this repository teaches.

```
LEVEL 0 — LLM Call
  A single stateless model invocation. No memory. No tools. No loops.
  Problem it cannot solve: anything requiring context, history, or iteration.

LEVEL 1 — Chain (LCEL)
  A deterministic sequence of LLM calls composed via the | pipe operator.
  No branching or cycles.
  Problem it cannot solve: tasks that require runtime decisions or tool use.

LEVEL 2 — Tool-Using Agent (ReAct)
  An LLM that selects and executes tools in a loop until it produces a final answer.
  Single actor. No persistent state between sessions.
  Problem it cannot solve: complex multi-step tasks with stateful context.

LEVEL 3 — Stateful Agent (LangGraph)
  A graph-based agent with a TypedDict State, conditional routing, and a
  MemorySaver checkpointer. Can pause, resume, and branch based on runtime state.
  Single actor.
  Problem it cannot solve: tasks that require specialised, parallel sub-workflows.

LEVEL 4 — Multi-Actor System (LangGraph)
  Multiple specialised agent nodes coordinated by a Supervisor or passed via
  Swarm handoff. Agents can delegate, parallelise, and share structured state.
  Problem it cannot solve: long-horizon tasks with planning, reflection, and memory.

LEVEL 5 — DeepAgent / Cognitive Architecture
  A multi-actor system augmented with:
    • Long-term memory (semantic + episodic via vector store)
    • Planning and task decomposition (Planner node)
    • Self-reflection and revision loops (Reflexion pattern)
    • Human-in-the-loop gates (interrupt + Command(resume=…))
    • Full LangSmith observability and structured logging
  This is the target architecture of the advanced curriculum.
```

---

## Prerequisites & Setup

### Required Tools

```bash
# Minimum Python version
python --version  # must be 3.11+

# Recommended: create a virtual environment per level or project
python -m venv .venv && source .venv/bin/activate

# Install a project's dependencies
pip install -r requirements.txt
```

### Required API Keys

Copy `.env.example` to `.env` in any project folder and fill in your keys:

```bash
cp .env.example .env
```

| Key | Required For | Get It At |
|---|---|---|
| `OPENAI_API_KEY` | All Level 1 examples; default LLM throughout | [platform.openai.com](https://platform.openai.com) |
| `ANTHROPIC_API_KEY` | Multi-model comparison projects | [console.anthropic.com](https://console.anthropic.com) |
| `GOOGLE_API_KEY` | Google Gemini model support | [aistudio.google.com](https://aistudio.google.com) |
| `LANGSMITH_API_KEY` | Tracing in all Level 2+ projects | [smith.langchain.com](https://smith.langchain.com) |
| `TAVILY_API_KEY` | Web search tools in Level 3 agents | [tavily.com](https://tavily.com) |
| `OPENWEATHERMAP_API_KEY` | Travel Planner project (optional) | [openweathermap.org](https://openweathermap.org/api) |

---

## Curriculum Overview

**20 projects total: 3 beginner · 5 intermediate · 12 advanced**

### Level 1 — Foundations `foundations/`

*Goal: understand the atomic building blocks of every LangChain application.*

| Module | Topic |
|---|---|
| [1.1 — Models, Prompts, Parsers](foundations/modules/module-1.1-what-is-langchain/README.md) | ChatOpenAI/Anthropic, ChatPromptTemplate, SystemMessage, StrOutputParser, PydanticOutputParser |
| [1.2 — LCEL and Runnables](foundations/modules/module-1.2-lcel-and-runnables/README.md) | `\|` pipe, RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, `.stream()`, `.batch()` |
| [1.3 — Structured Output](foundations/modules/module-1.3-output-formatting-and-structured-responses/README.md) | `.with_structured_output()`, Pydantic v2, JSON Schema, retry on parse failure |

| Project | Description | DeepAgent Level |
|---|---|---|
| [B-1 Smart Formatter CLI](foundations/projects/project-1.1-smart-formatter-cli/README.md) | Transform messy text into validated JSON via LCEL + PydanticOutputParser with retry loop | Level 1 |
| [B-2 Prompt Lab](foundations/projects/project-1.2-prompt-playground/README.md) | Run the same prompt across multiple LLMs simultaneously with RunnableParallel + Rich UI | Level 1 |
| [B-3 FAQ Generator](foundations/projects/project-1.3-faq-generator/README.md) | Ingest raw notes and produce structured, categorised FAQ entries with RunnableBranch | Level 1 |

---

### Level 2 — Context, Retrieval, and State `context-retrieval-state/`

*Goal: build applications that remember, retrieve, and reason over external information.*

| Module | Topic |
|---|---|
| [2.1 — Memory Fundamentals](context-retrieval-state/modules/module-2.1-memory-fundamentals/README.md) | Why LLMs are stateless; message list as memory primitive; BufferWindow vs Summary memory; RunnableWithMessageHistory |
| [2.2 — RAG End-to-End](context-retrieval-state/modules/module-2.2-rag-end-to-end/README.md) | Document loaders, RecursiveCharacterTextSplitter mechanics, embeddings geometry, FAISS/Chroma/Pinecone |
| [2.3 — Retrieval Quality](context-retrieval-state/modules/module-2.3-retrieval-quality/README.md) | MultiQueryRetriever, ContextualCompression, ParentDocument, LLM-as-judge, RAGAS evaluation |
| [2.4 — Stateful Applications](context-retrieval-state/modules/module-2.4-stateful-applications/README.md) | Session-scoped state, FileChatMessageHistory, user-context injection, multi-user isolation |

| Project | Description | DeepAgent Level |
|---|---|---|
| [I-1 DocuChat](context-retrieval-state/projects/project-2.1-docuchat/README.md) | Conversational PDF assistant with history-aware retriever and streaming Streamlit UI | Level 2 |
| [I-2 Knowledge Base Q&A](context-retrieval-state/projects/project-2.2-knowledge-base-qa/README.md) | Index a folder of docs; answer questions with exact source citations using Chroma + MMR | Level 2 |
| [I-3 Support Bot](context-retrieval-state/projects/project-2.3-support-bot/README.md) | Policy Q&A with grounding enforcement, confidence scoring, and HR escalation routing | Level 2 |
| [I-4 Meeting Notes Assistant](context-retrieval-state/projects/project-2.4-meeting-notes-assistant/README.md) | Extract decisions, action items, and owners from meeting transcripts into structured Pydantic records | Level 2 |
| [I-5 Research Digest Builder](context-retrieval-state/projects/project-2.5-research-digest-builder/README.md) | Ingest URLs, run 5 analyst queries in parallel via RunnableParallel, render Jinja2 newsletter | Level 2 |

---

### Level 3 — Tool Calling, Workflows & DeepAgents `deepagents/`

*Goal: move from single-response systems to durable, multi-step, autonomous workflows.*

| Module | Topic |
|---|---|
| [3.1 — Tool Calling Fundamentals](deepagents/modules/module-3.1-tool-calling-fundamentals/README.md) | API-level mechanics, `.bind_tools()`, `tool_choice`, ToolMessage parsing, error handling |
| [3.2 — Custom Tools](deepagents/modules/module-3.2-custom-tools/README.md) | `@tool` + Pydantic args_schema, async tools, ToolException, docstring quality |
| [3.3 — LangGraph Core](deepagents/modules/module-3.3-langgraph-basics/README.md) | StateGraph anatomy, TypedDict + Annotated reducers, nodes, conditional edges, MemorySaver, thread_id |
| [3.4 — Multi-Step Workflows](deepagents/modules/module-3.4-multi-step-agent-workflows/README.md) | ReAct loop from scratch, routing functions, `interrupt()`, `Command(resume=…)`, Send API fan-out |
| [3.5 — DeepAgents Architecture](deepagents/modules/module-3.5-deepagents-architecture/README.md) | Supervisor pattern, Swarm handoff, Plan-and-Execute, Reflexion, subgraph composition, loop guards |
| [3.6 — Observability & Debugging](deepagents/modules/module-3.6-observability-and-debugging/README.md) | LangSmith project setup, run metadata tagging, trace inspection, evaluator hooks, cost dashboards |

| Project | Description | Level |
|---|---|---|
| [A-1 Autonomous Research Assistant](deepagents/projects/project-3.1-autonomous-research-assistant/README.md) | Multi-step research: plan → parallel search → synthesise → detect contradictions → quality gate loop | 4–5 |
| [A-2 Multi-Tool Travel Planner](deepagents/projects/project-3.2-multi-tool-travel-planner/README.md) | Combine weather, maps, attractions, accommodation, and budget into a structured day-by-day itinerary | 3–4 |
| [A-3 Incident Triage Agent](deepagents/projects/project-3.3-incident-triage-agent/README.md) | Parse alerts, fetch logs/metrics, classify P1–P4, propose remediation, human gate for P1 escalation | 3–4 |
| [A-4 Customer Support Swarm](deepagents/projects/project-3.4-customer-support-triage-agent/README.md) | Peer-to-peer Swarm handoff routing across Billing, Tech, Account, Escalation specialist agents | 4 |
| [A-5 Codebase Explorer](deepagents/projects/project-3.5-codebase-explorer/README.md) | Scan a repo, infer architecture, generate Mermaid dependency diagram, answer developer questions | 3 |
| [A-6 Sales Intelligence Agent](deepagents/projects/project-3.6-sales-intelligence-agent/README.md) | Research a company → extract decision-makers, pain points, messaging angles → scored SalesBrief | 3–4 |
| [A-7 Meeting-to-Action Agent](deepagents/projects/project-3.7-meeting-to-action-agent/README.md) | Transcript → decisions + action items + follow-up email with human-in-the-loop review gate | 3 |
| [A-8 Data Query Agent](deepagents/projects/project-3.8-data-query-agent/README.md) | Natural language → validated SQL → execute → plain-English answer with self-correction loop | 3 |
| [A-9 Compliance Review Assistant](deepagents/projects/project-3.9-compliance-review-assistant/README.md) | Analyse a document against policy rules, flag violations with severity, generate risk report | 3–4 |
| [A-10 DeepAgents Orchestrator](deepagents/projects/project-3.10-deepagents-orchestrator/README.md) | **Capstone.** Supervisor orchestrates Researcher + Analyst + Writer + Reviewer with Reflexion loop | 5 |
| [A-11 Autonomous Content Ops](deepagents/projects/project-3.11-autonomous-content-ops-agent/README.md) | Brief → gather sources → draft → self-check brand guidelines → revise → publication-ready artifact | 5 |
| [A-12 Workflow Recovery Agent](deepagents/projects/project-3.12-workflow-recovery-agent/README.md) | Demonstrate production resilience: retries, fallbacks, checkpoint restore after simulated crash | 4 |

---

## Repository Structure

```text
LangChain-DeepAgents-Playbook/
├── README.md
├── .env.example                          ← master key template for all projects
├── shared/
│   └── utils/
│       ├── langsmith_setup.py            ← one-call LangSmith tracing bootstrap
│       ├── model_factory.py              ← provider-agnostic chat model factory
│       └── logging_config.py            ← structured logging with JSON mode
│
├── foundations/                          ← Level 1: Foundations
│   ├── modules/
│   │   ├── module-1.1-what-is-langchain/
│   │   ├── module-1.2-lcel-and-runnables/
│   │   └── module-1.3-output-formatting-and-structured-responses/
│   └── projects/
│       ├── project-1.1-smart-formatter-cli/
│       ├── project-1.2-prompt-playground/
│       └── project-1.3-faq-generator/
│
├── context-retrieval-state/              ← Level 2: Context, Retrieval, State
│   ├── modules/
│   │   ├── module-2.1-memory-fundamentals/
│   │   ├── module-2.2-rag-end-to-end/
│   │   ├── module-2.3-retrieval-quality/
│   │   └── module-2.4-stateful-applications/
│   └── projects/
│       ├── project-2.1-docuchat/
│       ├── project-2.2-knowledge-base-qa/
│       ├── project-2.3-support-bot/
│       ├── project-2.4-meeting-notes-assistant/
│       └── project-2.5-research-digest-builder/
│
└── deepagents/                           ← Level 3: Tools, Workflows, DeepAgents
    ├── modules/
    │   ├── module-3.1-tool-calling-fundamentals/
    │   ├── module-3.2-custom-tools/
    │   ├── module-3.3-langgraph-basics/
    │   ├── module-3.4-multi-step-agent-workflows/
    │   ├── module-3.5-deepagents-architecture/
    │   └── module-3.6-observability-and-debugging/
    └── projects/
        ├── project-3.1-autonomous-research-assistant/
        ├── project-3.2-multi-tool-travel-planner/
        … (through project-3.12-workflow-recovery-agent/)
```

---

## Modern Stack & Conventions

This repository uses **current LangChain architecture only**. Legacy patterns are explained as historical context, never taught as the default path.

| Component | Package | Why |
|---|---|---|
| Chat models | `langchain-openai`, `langchain-anthropic` | Provider-specific integrations |
| Chain composition | `langchain-core` LCEL `\|` pipe | Lazy eval, native streaming, type-safe composition |
| Structured output | `.with_structured_output()` + Pydantic v2 | Turns LLM responses into reliable software inputs |
| Stateful agents | `langgraph` StateGraph | TypedDict state, conditional routing, MemorySaver |
| Observability | LangSmith (all Level 2+ work) | Cannot debug what you cannot trace |
| Credentials | `python-dotenv` | Never hardcode API keys |

**What this repo deliberately avoids:**
- `from langchain.chains import ...` (legacy chains)
- `AgentExecutor` (replaced by LangGraph StateGraph)
- Legacy memory classes as primary patterns

---

## How to Use This Repo

### As a learner
Follow the levels sequentially. Do not skip Level 1 LCEL. Most confusion in later agent systems originates from a shaky understanding of composition and data flow.

### As a practitioner
Jump directly to any project. Each project README contains its own architecture diagram, design decisions, and setup instructions. The corresponding modules serve as reference material.

### As a team / for onboarding
Use the Level 2 RAG projects and Level 3 LangGraph projects as onboarding exercises. Each project is self-contained and can be run against a real codebase or document set.

### Running a project

```bash
# 1. Navigate to the project directory
cd deepagents/projects/project-3.1-autonomous-research-assistant

# 2. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Run the project entry point
python src/main.py --help
```

### Running tests

```bash
# From any advanced project directory
pip install pytest
pytest tests/ -v
```

---

## Contributing

Contributions that improve **clarity, correctness, examples, testing, and real-world applicability** are welcome.

### What makes a good contribution

- Fixing technical inaccuracies (wrong API, outdated import, broken code)
- Adding a missing pitfall or best practice to an existing module
- Improving inline comments or explanations without changing behavior
- Adding pytest cases to the `tests/` directory of any project
- Suggesting a new project that fits the curriculum structure

### What to avoid

- Introducing legacy patterns (`AgentExecutor`, `langchain.chains`, old-style memory)
- Adding features beyond what a module or project spec requires
- Removing the "why" explanations in favour of shorter code blocks

### Contribution workflow

```bash
# Fork the repo, then:
git checkout -b feature/your-contribution-name
# Make changes
git commit -m "docs: explain chunk_overlap trade-off in module 2.2"
git push origin feature/your-contribution-name
# Open a pull request
```

---

## License

MIT — see [LICENSE](LICENSE). Use freely, teach openly, build boldly.

---

*Built for developers who want to understand modern AI engineering — not just run demos.*
