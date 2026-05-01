[← Testing Tools in Isolation](04-tool-testing-in-isolation.md) | [Next Module → LangGraph Core](../../module-3.3-langgraph-basics/README.md)

---

# 05 — Tool Registry Pattern

## Why a Registry?

As a project grows, tools accumulate. A naive approach — a flat list of `@tool`-decorated
functions — breaks down when:

- Different agents need different tool subsets
- Tools need versioning when their interfaces change
- Permissions must be enforced (some users shouldn't have access to side-effect tools)
- You want to add or remove tools at runtime without restarting the process

A **tool registry** treats tools as first-class managed objects with metadata, grouping,
and runtime selection.

---

## Real-World Analogy

Think of a hospital's medication cabinet. Every drug is registered with its name,
dosage form, and who is authorised to prescribe it. A general practitioner can prescribe
common medications; a specialist can prescribe controlled substances; a nurse can only
administer, not prescribe. The cabinet doesn't hand out drugs based on a flat list —
it enforces roles.

A tool registry does the same for agents: it knows which tools exist, what role they
belong to, and whether the current agent has permission to use them.

---

## Basic Registry Structure

```python
# tools/registry.py
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.tools import BaseTool

@dataclass
class ToolEntry:
    """Metadata container for a registered tool."""
    tool: BaseTool
    group: str                   # e.g. "research", "write", "admin"
    version: str = "1.0.0"
    description: str = ""        # human-readable; may differ from tool.description
    requires_approval: bool = False   # True for side-effect tools
    allowed_roles: list[str] = field(default_factory=lambda: ["*"])  # "*" = all roles

class ToolRegistry:
    """
    Central registry for all tools in the system.
    Supports grouping, role-based access, and versioning.
    """

    def __init__(self):
        self._registry: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> "ToolRegistry":
        """Register a tool. Returns self for chaining."""
        name = entry.tool.name
        if name in self._registry:
            raise ValueError(f"Tool '{name}' is already registered. Use register_version() to add a new version.")
        self._registry[name] = entry
        return self

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        entry = self._registry.get(name)
        return entry.tool if entry else None

    def get_for_role(self, role: str) -> list[BaseTool]:
        """Return all tools accessible to the given role."""
        result = []
        for entry in self._registry.values():
            if "*" in entry.allowed_roles or role in entry.allowed_roles:
                result.append(entry.tool)
        return result

    def get_by_group(self, group: str) -> list[BaseTool]:
        """Return all tools in a specific group."""
        return [e.tool for e in self._registry.values() if e.group == group]

    def get_safe_tools(self) -> list[BaseTool]:
        """Return tools that do not require human approval."""
        return [e.tool for e in self._registry.values() if not e.requires_approval]

    def get_all(self) -> list[BaseTool]:
        """Return all registered tools."""
        return [e.tool for e in self._registry.values()]

    def list_names(self) -> list[str]:
        return list(self._registry.keys())

    def __repr__(self) -> str:
        lines = ["ToolRegistry:"]
        for name, entry in self._registry.items():
            approval = " [approval required]" if entry.requires_approval else ""
            lines.append(f"  {name} (group={entry.group}, v{entry.version}){approval}")
        return "\n".join(lines)
```

---

## Populating the Registry

```python
# tools/definitions/research.py
from langchain_core.tools import tool
import json

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Returns JSON array of {title, url, snippet}."""
    return json.dumps([{"title": "Result", "url": "https://example.com", "snippet": "..."}])

@tool
def fetch_arxiv_paper(arxiv_id: str) -> str:
    """
    Fetch the abstract and metadata of an arXiv paper by its ID (e.g. '2310.01234').
    Returns JSON with title, authors, abstract, published_date.
    """
    return json.dumps({"title": "Paper Title", "abstract": "Abstract text...", "authors": []})

# tools/definitions/write.py
@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. Returns 'success' or an error message."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return "success"
    except Exception as e:
        return f"error: {e}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the given address. Returns 'sent' or error message."""
    # Stub: real implementation uses SMTP or SendGrid
    return f"sent: email to {to} with subject '{subject}'"
```

```python
# tools/setup.py — build the registry at application startup
from tools.registry import ToolRegistry, ToolEntry
from tools.definitions.research import search_web, fetch_arxiv_paper
from tools.definitions.write import write_file, send_email

registry = ToolRegistry()

registry.register(ToolEntry(
    tool=search_web,
    group="research",
    version="1.2.0",
    allowed_roles=["*"],
    requires_approval=False,
))

registry.register(ToolEntry(
    tool=fetch_arxiv_paper,
    group="research",
    version="1.0.0",
    allowed_roles=["researcher", "analyst"],
    requires_approval=False,
))

registry.register(ToolEntry(
    tool=write_file,
    group="write",
    version="2.0.0",
    allowed_roles=["writer", "admin"],
    requires_approval=False,
))

registry.register(ToolEntry(
    tool=send_email,
    group="write",
    version="1.1.0",
    allowed_roles=["manager", "admin"],
    requires_approval=True,   # Requires human approval gate
))

print(registry)
# ToolRegistry:
#   search_web (group=research, v1.2.0)
#   fetch_arxiv_paper (group=research, v1.0.0)
#   write_file (group=write, v2.0.0)
#   send_email (group=write, v1.1.0) [approval required]
```

---

## Runtime Tool Selection for an Agent

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def build_agent_for_role(role: str):
    """Build an LLM instance bound to the tools appropriate for this role."""
    tools = registry.get_for_role(role)
    if not tools:
        raise ValueError(f"No tools found for role '{role}'")
    return llm.bind_tools(tools, tool_choice="auto")

# Research agent — gets search_web and fetch_arxiv_paper:
researcher_llm = build_agent_for_role("researcher")

# Manager agent — gets search_web and send_email:
manager_llm = build_agent_for_role("manager")

# Admin agent — gets everything:
admin_llm = build_agent_for_role("admin")
```

---

## Tool Versioning — Replacing a Tool Without Breaking Existing Agents

```python
class ToolRegistry:
    # ... (existing code) ...

    def register_version(self, entry: ToolEntry, replace: bool = False) -> "ToolRegistry":
        """
        Register a new version of an existing tool.
        If replace=True, the old version is overwritten in the active registry.
        If replace=False, the new version is stored alongside the old under a versioned name.
        """
        name = entry.tool.name
        if replace:
            self._registry[name] = entry
        else:
            versioned_name = f"{name}_v{entry.version.replace('.', '_')}"
            # Register under versioned name — old version remains active
            self._registry[versioned_name] = entry
        return self
```

---

## Per-User Tool Permissions

For multi-user systems, extend the registry lookup to check user context:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class UserContext:
    user_id: str
    role: str
    feature_flags: list[str] = None

def get_tools_for_user(registry: ToolRegistry, user: UserContext) -> list:
    """
    Return the tool list for a specific user, respecting role permissions
    and feature flag overrides.
    """
    base_tools = registry.get_for_role(user.role)

    # Feature flag override: give beta users access to experimental tools
    if user.feature_flags and "beta_tools" in user.feature_flags:
        base_tools += registry.get_by_group("beta")

    return base_tools

# Usage:
alice = UserContext(user_id="u-001", role="researcher", feature_flags=["beta_tools"])
alice_tools = get_tools_for_user(registry, alice)
alice_llm = llm.bind_tools(alice_tools)
```

---

## Tool Retrieval for Large Registries

When the registry has 20+ tools, passing all of them to the model degrades selection
accuracy. Use semantic retrieval to select the most relevant tools per query:

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

class SemanticToolSelector:
    """
    Selects the top-K most relevant tools from the registry for a given user query,
    using semantic similarity on tool descriptions.
    """

    def __init__(self, registry: ToolRegistry, top_k: int = 8):
        self.top_k = top_k
        self._tools = registry.get_all()
        embeddings = OpenAIEmbeddings()
        self._store = InMemoryVectorStore(embedding=embeddings)
        texts = [f"{t.name}: {t.description}" for t in self._tools]
        self._store.add_texts(texts=texts)

    def select(self, query: str) -> list:
        results = self._store.similarity_search(query, k=self.top_k)
        selected_names = {r.page_content.split(":")[0].strip() for r in results}
        return [t for t in self._tools if t.name in selected_names]

# Usage:
selector = SemanticToolSelector(registry, top_k=6)

def dynamic_agent(query: str):
    relevant_tools = selector.select(query)
    llm_with_tools = llm.bind_tools(relevant_tools)
    return llm_with_tools.invoke(query)
```

---

## Common Pitfalls

| Pitfall                                             | Symptom                                        | Fix                                                                                 |
| --------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| Flat list grows to 30+ tools                        | Model picks wrong tool ~25% of the time        | Use the registry + semantic selector                                                |
| No versioning strategy                              | Updating a tool breaks existing agents mid-run | Use `register_version()` with `replace=False` to keep old versions active           |
| Per-user permissions ignored at invocation          | Users execute tools they shouldn't have        | Enforce permissions both at bind time AND at invocation (double-check in tool node) |
| Registry built at import time with I/O side effects | App startup fails or slows due to DB/API calls | Only register tool metadata at startup; don't call tools at registration            |
| No test for the registry itself                     | Misconfiguration not caught until runtime      | Unit-test `registry.get_for_role()` and `registry.get_by_group()`                   |

---

## Mini Summary

- A tool registry centralises tool metadata: group, version, required approval, allowed roles
- `get_for_role(role)` returns only the tools a given agent is permitted to use
- `get_by_group(group)` organises tools by functional category (research, write, admin)
- For 20+ tools, add a semantic selector to avoid passing the full registry to every model call
- Version tools explicitly; use `replace=False` to keep old versions active while testing new ones

---

[← Testing Tools in Isolation](04-tool-testing-in-isolation.md) | [Next Module → LangGraph Core](../../module-3.3-langgraph-basics/README.md)
