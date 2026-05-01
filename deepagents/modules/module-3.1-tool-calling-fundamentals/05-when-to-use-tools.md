[← Tool Error Handling](04-tool-error-handling.md) | [Next Module → Custom Tools](../../module-3.2-custom-tools/README.md)

---

# 05 — When to Use Tools

## The Problem This File Solves

Most agents are over-tooled. Developers add a tool for every possible action and then
wonder why the model calls the wrong one 30% of the time. Tools are not free: each one
adds tokens to the prompt, increases the surface area for misprediction, and introduces
a new failure mode. The right question is not "can this be a tool?" but "should this be a tool?"

---

## Real-World Analogy

A Swiss Army knife with 47 blades is impressive but slow to navigate.
A professional chef carries exactly three knives — each one specific, each one irreplaceable.
The chef doesn't add a fourth knife until the first three provably can't do the job.

Design your tool set like the chef's knife roll, not like the Swiss Army knife.

---

## Decision Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│           IS A TOOL THE RIGHT SOLUTION?                                 │
│                                                                         │
│  Question 1: Does it need live or dynamic data?                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  NO  → Does the LLM already know this? Yes → Use a prompt.              │
│        Can a chain transform/format it? Yes → Use LCEL.                 │
│  YES → Continue to Question 2.                                          │
│                                                                         │
│  Question 2: Does it produce side effects?                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  YES → Is the side effect reversible? No → Require human approval gate  │
│        (Module 3.4 human-in-the-loop)                                   │
│  NO  → Read-only tool; safe to call without approval.                   │
│                                                                         │
│  Question 3: Does it require structured external interaction?           │
│  ─────────────────────────────────────────────────────────────────────  │
│  YES → Tool is appropriate. Go to naming step.                          │
│  NO  → Revisit: can the model reason about this internally?             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tool vs Prompt Engineering vs Chain

| Approach                     | When to use                                 | Example                                        |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------- |
| **Prompt engineering**       | LLM has the knowledge; just needs framing   | "Translate this text to French"                |
| **LCEL chain**               | Deterministic transformation on fixed input | "Parse JSON → validate schema → format output" |
| **Tool (read-only)**         | Needs live or private data at runtime       | Web search, database query, calendar lookup    |
| **Tool (write/side-effect)** | Must take action in the external world      | Send email, create ticket, write file          |

Misclassification example:

> "I want the model to summarize a document"
> → Wrong: `@tool def summarize(text: str)` (just calls the LLM inside a tool — pointless)
> → Right: Chain: `load_document | summarize_prompt | llm | StrOutputParser()`

---

## Read-Only vs Side-Effect Tools

This distinction matters enormously for safety and testing:

```
Read-Only Tools (safe to auto-execute)       Side-Effect Tools (require review)
───────────────────────────────────────      ─────────────────────────────────────
search_web(query)                            send_email(to, subject, body)
get_stock_price(ticker)                      create_calendar_event(time, attendees)
fetch_user_profile(user_id)                  delete_record(table, id)
lookup_policy(topic)                         post_to_slack(channel, message)
calculate_distance(a, b)                     push_to_github(repo, branch, commit)
read_file(path)                              write_file(path, content)
```

**Pattern:** Group these into separate tool lists and bind them to different agents
(or different pipeline stages). The _research_ stage uses only read-only tools;
the _action_ stage uses side-effect tools with a human approval gate.

---

## Tool Scope and Single Responsibility

Every tool should do exactly one thing. The model chooses tools based on name and description.
A tool that does too much confuses the model's choice:

```python
# ❌ BAD — too broad, unclear when to call
@tool
def manage_customer(action: str, customer_id: str, data: dict = None) -> str:
    """Manage a customer: fetch, update, delete, or send email."""
    if action == "fetch":   ...
    elif action == "update": ...
    elif action == "delete": ...
    elif action == "email":  ...

# ✅ GOOD — single responsibility, unambiguous when to call each one
@tool
def get_customer(customer_id: str) -> str:
    """Retrieve a customer record by ID. Returns customer name, email, and plan."""
    ...

@tool
def update_customer_plan(customer_id: str, new_plan: str) -> str:
    """Change a customer's subscription plan. new_plan must be 'free', 'pro', or 'enterprise'."""
    ...

@tool
def send_customer_email(customer_id: str, subject: str, body: str) -> str:
    """Send a transactional email to a customer. Subject must be under 100 characters."""
    ...
```

---

## Tool Naming Conventions That Help the Model

The model uses the tool's **name** as the primary signal for selection.
Follow these conventions to maximise correct tool selection:

```
verb_noun pattern:
  search_web          ✅   web_search_tool     ❌
  get_weather         ✅   weather             ❌
  send_email          ✅   emailer             ❌
  create_ticket       ✅   ticket_creation     ❌

Be specific about the noun:
  get_customer_profile  ✅   get_info            ❌
  search_knowledge_base ✅   search              ❌  (ambiguous)
  calculate_tax_rate    ✅   calculate           ❌

Avoid abbreviations:
  get_stock_price  ✅   gsp          ❌
  fetch_user       ✅   usr_fetch    ❌
```

### Docstring Quality — Bad vs Good

The model reads the docstring to decide _whether_ to call the tool and _how_ to fill arguments.

```python
# ❌ BAD DOCSTRING — tells the model almost nothing
@tool
def get_weather(city: str) -> str:
    """Get weather."""
    ...

# ✅ GOOD DOCSTRING — describes output, argument format, and when to use it
@tool
def get_weather(city: str) -> str:
    """
    Retrieve current weather conditions for a city.
    Returns a JSON string with keys: temperature_celsius, condition, humidity_percent.
    Use this when the user asks about current or today's weather.
    city should be a proper city name (e.g., 'Paris', 'New York', 'Tokyo').
    Do NOT use for historical weather or forecasts beyond today.
    """
    ...
```

---

## How Many Tools is Too Many?

Empirical benchmarks on instruction-following models suggest:

- **1–5 tools:** Near-perfect selection accuracy
- **6–15 tools:** Good accuracy with well-named, distinct tools
- **16–30 tools:** Noticeable degradation; model may choose wrong tool ~10–25% of the time
- **30+ tools:** Use a tool retrieval layer (RAG over tool descriptions) rather than passing all at once

```python
# Tool retrieval pattern for large tool sets:
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

def build_tool_retriever(tools, top_k=8):
    """Return a function that retrieves the top_k most relevant tools for a query."""
    embeddings = OpenAIEmbeddings()
    store = InMemoryVectorStore(embedding=embeddings)
    # Store tool descriptions as documents
    documents = [
        {"page_content": f"{t.name}: {t.description}", "metadata": {"tool": t}}
        for t in tools
    ]
    store.add_texts(
        texts=[d["page_content"] for d in documents],
        metadatas=[d["metadata"] for d in documents],
    )
    def retrieve(query: str) -> list:
        results = store.similarity_search(query, k=top_k)
        return [r.metadata["tool"] for r in results]
    return retrieve

# Usage:
# tool_retriever = build_tool_retriever(ALL_TOOLS)
# relevant_tools = tool_retriever(user_query)
# llm_with_tools = llm.bind_tools(relevant_tools)
```

---

## Tool Responsibility Matrix

Use this template when designing a multi-agent system:

```
┌──────────────────────┬────────────────────────────────┬─────────────────────┐
│ Agent                │ Read-only tools                │ Side-effect tools   │
├──────────────────────┼────────────────────────────────┼─────────────────────┤
│ Researcher           │ search_web, fetch_paper        │ (none)              │
│ Analyst              │ query_db, calculate_metrics    │ (none)              │
│ Writer               │ read_file, get_brand_guidelines│ write_file          │
│ Publisher (gated)    │ (none)                         │ post_article,       │
│                      │                                │ send_newsletter     │
│ Admin (human-gated)  │ (none)                         │ delete_records,     │
│                      │                                │ update_permissions  │
└──────────────────────┴────────────────────────────────┴─────────────────────┘
```

The human approval gate (Module 3.4) sits between the Writer and Publisher, and
between the Analyst and Admin agents.

---

## Common Pitfalls

| Pitfall                                       | Symptom                                        | Fix                                                                |
| --------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------ |
| Wrapping an LLM call as a tool                | Nested LLM calls, unpredictable cost           | Use a sub-chain or sub-graph instead                               |
| Too many tools with overlapping descriptions  | Model picks the wrong tool ~25% of the time    | Merge overlapping tools; use tool retrieval for large sets         |
| Tool with side effects bound in "auto" mode   | Accidental data writes or emails sent          | Require explicit `tool_choice` or human gate for side-effect tools |
| Generic names like `helper`, `process`, `run` | Model never calls it (can't infer when to use) | Follow `verb_noun` naming; be specific                             |
| Stateful tools with no isolation              | Tool from user A affects user B's data         | Pass user_id as a parameter; tools should be stateless             |

---

## Mini Summary

- Tools are appropriate for live/private data and external actions — not for knowledge the LLM already has
- Separate read-only tools (safe to auto-execute) from side-effect tools (require approval gates)
- Single-responsibility: one tool, one job — makes model selection accurate
- Naming: `verb_noun` pattern, no abbreviations, no generic names
- Docstrings are part of the prompt: describe output format, expected argument values, and when NOT to use the tool
- For 15+ tools, use a retrieval layer to pass only the relevant subset per query

---

[← Tool Error Handling](04-tool-error-handling.md) | [Next Module → Custom Tools](../../module-3.2-custom-tools/README.md)
