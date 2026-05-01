[← Swarm Handoff](03-swarm-handoff-pattern.md) | [Next → Reflexion Pattern](05-reflexion-pattern.md)

---

# 04 — Plan and Execute

## Why Planning Beats Pure ReAct for Complex Tasks

A pure ReAct agent is reactive: it sees the current state, picks the next best tool,
and repeats. This works well for tasks where the right next step is always obvious
from the current context. But for tasks with many steps — especially when early steps
affect which later steps are even relevant — pure ReAct produces two failure modes:

1. **Myopic decisions**: The agent optimises each step locally without considering
   the full plan. It may spend 5 steps researching a dead end.
2. **Context loss**: Each tool call adds to the message history. By step 15, the
   original goal is buried under 2,000 tokens of intermediate results.

Plan and Execute separates thinking from doing:

- The **Planner** thinks once, produces a full structured plan
- The **Executor** does, working through the plan step by step
- The **Updater** revises the plan when results change what's needed

---

## Real-World Analogy

A software project. The architect (Planner) creates the technical design before any
code is written. Developers (Executor) implement one story at a time. If a story
reveals a constraint that changes the design, the architect updates the spec before
the team picks up the next story. Coding while the architect is still designing is
chaos. Designing during coding is rework.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PLANNER                                 │
│  Input: goal + any prior context                                │
│  Output: TaskPlan (list of TaskStep objects)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ TaskPlan
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         EXECUTOR                                │
│  Picks next incomplete TaskStep → runs tools → stores result    │
│  Marks step as complete                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ step result
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PLAN UPDATER (optional)                      │
│  Reviews result → may add/modify/remove future steps            │
│  Marks whether plan is complete                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  complete?  │
                    └──────┬──────┘
              Yes ◄────────┴─────────► No
              │                        │
              ▼                        ▼
         SYNTHESISER              (loop back to EXECUTOR)
```

---

## Pydantic TaskPlan Model

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    SKIPPED   = "skipped"   # plan update determined this step is no longer needed

class TaskStep(BaseModel):
    step_id:     int
    description: str
    tool_hint:   Optional[str] = None  # which tool is likely needed
    result:      Optional[str] = None  # filled in by Executor
    status:      StepStatus = StepStatus.PENDING

class TaskPlan(BaseModel):
    goal:        str
    steps:       list[TaskStep]
    is_complete: bool = False

    def next_pending_step(self) -> Optional[TaskStep]:
        return next((s for s in self.steps if s.status == StepStatus.PENDING), None)

    def all_done(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)
```

---

## State Design

```python
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class PlanExecuteState(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    goal:        str
    plan:        Optional[dict]   # serialised TaskPlan
    results:     Annotated[list[str], operator.add]
    final_answer: Optional[str]
    iteration:   int
```

Why serialise `TaskPlan` as `dict`? TypedDict values must be JSON-serialisable for
MemorySaver checkpointing. Store the model as `.model_dump()` and reconstruct with
`TaskPlan.model_validate(state["plan"])` when needed.

---

## Planner Node

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)   # use smarter model for planning

def planner_node(state: PlanExecuteState) -> dict:
    """
    Creates a structured task plan for the goal.
    Only runs once — if a plan already exists, skip planning.
    """
    if state.get("plan") is not None:
        return {}   # plan already exists — nothing to update

    system = SystemMessage(
        "You are a strategic planner. Given a goal, create a detailed step-by-step plan.\n"
        "Output ONLY valid JSON matching this schema:\n"
        '{"goal": "<goal>", "steps": [{"step_id": 1, "description": "<what to do>", '
        '"tool_hint": "<optional: tool name>", "result": null, "status": "pending"}, ...], '
        '"is_complete": false}\n'
        "Create 3-7 concrete, actionable steps. Be specific."
    )
    human = HumanMessage(f"Goal: {state['goal']}")

    response = planner_llm.invoke([system, human])

    try:
        plan_dict = json.loads(response.content)
        TaskPlan.model_validate(plan_dict)  # validate schema
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: single-step plan
        plan_dict = TaskPlan(
            goal=state["goal"],
            steps=[TaskStep(step_id=1, description="Research and answer the goal directly.")]
        ).model_dump()

    return {
        "plan":     plan_dict,
        "messages": [AIMessage(content=f"Plan created with {len(plan_dict['steps'])} steps.")],
    }
```

---

## Executor Node

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[MOCK RESULT for '{query}': relevant information found]"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

executor_tools = [search_web, calculate]
executor_llm = llm.bind_tools(executor_tools)

def executor_node(state: PlanExecuteState) -> dict:
    """
    Picks the next pending step from the plan, executes it with tools,
    and marks it as done.
    """
    plan = TaskPlan.model_validate(state["plan"])
    next_step = plan.next_pending_step()

    if next_step is None:
        # All steps done — signal completion
        plan.is_complete = True
        return {"plan": plan.model_dump()}

    # Mark step as running
    next_step.status = StepStatus.RUNNING

    system = SystemMessage(
        f"You are executing step {next_step.step_id} of a multi-step plan.\n"
        f"Overall goal: {plan.goal}\n"
        f"Current step: {next_step.description}\n"
        "Use the available tools to complete this step. "
        "Provide a clear, concise result when done."
    )
    response = executor_llm.invoke([system, *state["messages"]])

    # Simulate tool execution result (in real graph, ToolNode handles this)
    result_text = (
        response.tool_calls[0]["args"].get("query", "result")
        if response.tool_calls
        else response.content
    )

    # Update step in plan
    next_step.result = result_text
    next_step.status = StepStatus.DONE

    return {
        "plan":      plan.model_dump(),
        "results":   [f"Step {next_step.step_id}: {result_text}"],
        "messages":  [response, AIMessage(content=f"Step {next_step.step_id} complete.")],
        "iteration": state["iteration"] + 1,
    }
```

---

## Plan Updater Node (Optional)

```python
def plan_updater_node(state: PlanExecuteState) -> dict:
    """
    Reviews the latest result and optionally modifies future steps.
    This is what makes Plan+Execute adaptive rather than rigid.
    """
    plan = TaskPlan.model_validate(state["plan"])

    # Only remaining pending steps are worth updating
    pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
    if not pending:
        plan.is_complete = True
        return {"plan": plan.model_dump()}

    latest_result = state["results"][-1] if state.get("results") else ""

    system = SystemMessage(
        "You are reviewing progress on a multi-step plan.\n"
        "Based on the latest result, decide if any remaining steps need to change.\n"
        "You may: update step descriptions, skip irrelevant steps (status='skipped'), "
        "or leave them unchanged.\n"
        "Respond ONLY with the updated 'steps' list as JSON, or 'NO_CHANGE' if nothing changes."
    )
    human = HumanMessage(
        f"Goal: {plan.goal}\n"
        f"Latest result: {latest_result}\n"
        f"Remaining steps: {[s.model_dump() for s in pending]}"
    )
    response = llm.invoke([system, human])

    if response.content.strip() != "NO_CHANGE":
        try:
            updated_steps_raw = json.loads(response.content)
            # Merge updates into plan
            step_map = {s.step_id: s for s in plan.steps}
            for updated in updated_steps_raw:
                if updated["step_id"] in step_map:
                    existing = step_map[updated["step_id"]]
                    existing.description = updated.get("description", existing.description)
                    existing.status = StepStatus(updated.get("status", existing.status))
        except Exception:
            pass   # keep plan unchanged on parse failure

    return {"plan": plan.model_dump()}
```

---

## Graph Assembly

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

MAX_EXECUTOR_ITERATIONS = 15

def route_after_executor(state: PlanExecuteState) -> str:
    """Route based on plan completion and iteration count."""
    if state["iteration"] >= MAX_EXECUTOR_ITERATIONS:
        return "synthesiser"

    plan = TaskPlan.model_validate(state["plan"])
    if plan.all_done() or plan.is_complete:
        return "synthesiser"

    return "plan_updater"   # review plan, then loop back to executor

def synthesiser_node(state: PlanExecuteState) -> dict:
    all_results = "\n\n".join(state.get("results", []))
    system = SystemMessage(
        "Synthesise all the research results into a final, coherent answer."
    )
    human = HumanMessage(
        f"Goal: {state['goal']}\n\nAll results:\n{all_results}"
    )
    response = llm.invoke([system, human])
    return {"final_answer": response.content}

builder = StateGraph(PlanExecuteState)
builder.add_node("planner",      planner_node)
builder.add_node("executor",     executor_node)
builder.add_node("plan_updater", plan_updater_node)
builder.add_node("synthesiser",  synthesiser_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_conditional_edges(
    "executor",
    route_after_executor,
    {"plan_updater": "plan_updater", "synthesiser": "synthesiser"},
)
builder.add_edge("plan_updater", "executor")   # loop: update → execute → update...
builder.add_edge("synthesiser", END)

graph = builder.compile(checkpointer=MemorySaver())
```

---

## Why Planning Beats Pure ReAct for Multi-Day Tasks

```
Pure ReAct for a 3-day research task:
  - Day 1: Model has full context; picks good steps
  - Day 2: Context window is 60% full of intermediate results
  - Day 3: Model loses track of original goal; starts repeating work
  - Result: Incomplete, redundant output

Plan + Execute for the same task:
  - Day 1: Planner creates 15-step plan; persisted in checkpointed state
  - Day 2: Executor picks up exactly where it left off (step 6 of 15)
  - Day 3: Plan Updater discards irrelevant steps; Executor finishes efficiently
  - Result: Complete, structured output in fewer total LLM calls
```

---

## Common Pitfalls

| Pitfall                                      | Symptom                                         | Fix                                                                                   |
| -------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| Re-running Planner on every executor loop    | New plan on every step; prior steps overwritten | Check `if state["plan"] is not None: return {}` in Planner                            |
| `TaskPlan` not serialised as `dict` in state | MemorySaver checkpointing fails                 | Use `plan.model_dump()` when updating state; `TaskPlan.model_validate()` when reading |
| No MAX_EXECUTOR_ITERATIONS                   | Executor loops forever on an open-ended plan    | Add iteration counter; route to synthesiser at threshold                              |
| Plan Updater changes already-completed steps | Completed work gets overwritten                 | Updater must only modify steps with `status == "pending"`                             |
| Executor tries to run all steps in one call  | Step results overwrite each other               | Executor must pick exactly ONE pending step per invocation                            |

---

## Mini Summary

- Plan and Execute separates thinking (Planner, once) from doing (Executor, loop)
- `TaskPlan` with `TaskStep` objects tracked in state enables progress persistence across restarts
- The Plan Updater makes the pattern adaptive: it revises remaining steps based on what was learned
- Always serialise the plan as a `dict` in state; validate back to the Pydantic model inside nodes
- Add `MAX_EXECUTOR_ITERATIONS` to prevent unbounded loops on poorly scoped tasks

---

[← Swarm Handoff](03-swarm-handoff-pattern.md) | [Next → Reflexion Pattern](05-reflexion-pattern.md)
