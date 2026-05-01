from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .config import Settings


RUNBOOKS = {
    "api": "API Error Spike Runbook: check upstream dependency health, inspect recent deploys, and confirm error budget impact.",
    "latency": "Latency Degradation Runbook: compare p95 by region, check queue depth, and inspect database saturation.",
}


def assess_severity(title: str, summary: str) -> Literal["sev3", "sev2", "sev1"]:
    combined = f"{title} {summary}".lower()
    if "checkout" in combined or "outage" in combined:
        return "sev1"
    if "spike" in combined or "timeout" in combined:
        return "sev2"
    return "sev3"


def lookup_runbook(title: str, summary: str) -> str:
    combined = f"{title} {summary}".lower()
    if "latency" in combined:
        return RUNBOOKS["latency"]
    return RUNBOOKS["api"]


class TriageDecision(BaseModel):
    severity: Literal["sev3", "sev2", "sev1"] = Field(..., description="Operational severity tier.")
    likely_cause: str = Field(..., description="Most plausible explanation based on the alert summary and runbook.")
    recommended_actions: list[str] = Field(default_factory=list, description="Immediate operational next steps.")
    runbook_used: str = Field(..., description="Name or description of the runbook referenced.")


class IncidentState(TypedDict, total=False):
    title: str
    summary: str
    context_note: str
    severity: Literal["sev3", "sev2", "sev1"]
    runbook: str
    root_cause_hypothesis: str
    remediation_steps: Annotated[list[str], operator.add]
    runbook_used: str
    escalation_approved: bool
    final_decision: dict


def escalation_gate(state: IncidentState) -> Command:
    """Pause for human approval on sev1 incidents with no clear remediation."""
    if state["severity"] == "sev1" and not state.get("remediation_steps"):
        human_decision = interrupt(
            {
                "message": "sev1 incident with no automated remediation. Approve escalation?",
                "triage_so_far": state["root_cause_hypothesis"],
            }
        )
        return Command(
            update={"escalation_approved": human_decision.get("approved", False)},
            goto="write_report",
        )
    return Command(update={"escalation_approved": False}, goto="write_report")


def write_report(state: IncidentState) -> IncidentState:
    return {
        "final_decision": {
            "severity": state["severity"],
            "likely_cause": state["root_cause_hypothesis"],
            "recommended_actions": state.get("remediation_steps", []),
            "runbook_used": state["runbook_used"],
            "escalation_approved": state.get("escalation_approved", False),
        }
    }


def build_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You produce concise incident triage decisions based on alert context and runbooks."),
            (
                "human",
                "Alert title: {title}\nAlert summary: {summary}\nContext note: {context_note}\nSeverity: {severity}\nRunbook: {runbook}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    decider = prompt | model.with_structured_output(TriageDecision)

    def context_node(state: IncidentState) -> IncidentState:
        return {"context_note": f"Recent alert inspection suggests this is affecting the primary API path for: {state['title']}"}

    def severity_node(state: IncidentState) -> IncidentState:
        return {"severity": assess_severity(state["title"], state["summary"])}

    def runbook_node(state: IncidentState) -> IncidentState:
        return {"runbook": lookup_runbook(state["title"], state["summary"])}

    def decision_node(state: IncidentState) -> IncidentState:
        decision = decider.invoke(
            {
                "title": state["title"],
                "summary": state["summary"],
                "context_note": state["context_note"],
                "severity": state["severity"],
                "runbook": state["runbook"],
            }
        )
        return {
            "root_cause_hypothesis": decision.likely_cause,
            "remediation_steps": decision.recommended_actions,
            "runbook_used": decision.runbook_used,
        }

    graph = StateGraph(IncidentState)
    graph.add_node("context", context_node)
    graph.add_node("severity", severity_node)
    graph.add_node("runbook", runbook_node)
    graph.add_node("decision", decision_node)
    graph.add_node("escalation_gate", escalation_gate)
    graph.add_node("write_report", write_report)
    graph.add_edge(START, "context")
    graph.add_edge("context", "severity")
    graph.add_edge("severity", "runbook")
    graph.add_edge("runbook", "decision")
    graph.add_edge("decision", "escalation_gate")
    graph.add_edge("write_report", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
