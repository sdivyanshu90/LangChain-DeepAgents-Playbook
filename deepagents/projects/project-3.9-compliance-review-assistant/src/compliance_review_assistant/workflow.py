from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings


POLICIES = {
    "vendor_access": "Checklist: security questionnaire completed, data access scope defined, production access approval documented.",
    "launch_review": "Checklist: rollback plan documented, owner assigned, customer communication plan prepared.",
}

REQUIRED_SECTIONS = {
    "vendor_access": ["security questionnaire", "data access scope", "production access approval"],
    "launch_review": ["rollback plan", "owner", "customer communication plan"],
}


def lookup_policy(policy_topic: str) -> str:
    return POLICIES.get(policy_topic, "No policy checklist found for this topic.")


class ComplianceReport(BaseModel):
    summary: str = Field(..., description="Short review summary.")
    risks: list[str] = Field(default_factory=list, description="Detected policy or process risks.")
    missing_information: list[str] = Field(default_factory=list, description="Required details that are not present in the submission.")
    recommendations: list[str] = Field(default_factory=list, description="Specific remediation recommendations.")
    risk_score: int = Field(..., description="Risk score from 0 to 100.")
    escalation_required: bool = Field(..., description="Whether human review should be required.")


class PolicyViolation(BaseModel):
    description: str = Field(..., description="Detected policy issue.")
    severity: Literal["critical", "major", "minor"] = Field(..., description="Violation severity tier.")


class ViolationAnalysis(BaseModel):
    summary: str = Field(..., description="Short compliance summary.")
    violations: list[PolicyViolation] = Field(default_factory=list, description="Detected policy violations.")
    escalation_required: bool = Field(default=False, description="Whether the case should be escalated.")


class MissingSectionsResult(BaseModel):
    missing_sections: list[str] = Field(default_factory=list, description="Required sections absent from the submission.")


class ComplianceState(TypedDict, total=False):
    policy_topic: str
    submission: str
    policy_text: str
    document_type: str
    document_text: str
    summary: str
    violations: Annotated[list[dict], operator.add]
    missing_sections: Annotated[list[str], operator.add]
    risk_score: int
    recommendations: Annotated[list[str], operator.add]
    escalation_required: bool
    report: dict


def required_sections_for(document_type: str) -> list[str]:
    return REQUIRED_SECTIONS.get(document_type, [])


def calculate_risk_score(violations: list[dict], missing_sections: list[str]) -> int:
    critical = sum(1 for violation in violations if violation.get("severity") == "critical")
    major = sum(1 for violation in violations if violation.get("severity") == "major")
    minor = sum(1 for violation in violations if violation.get("severity") == "minor")
    score = (critical * 30) + (major * 10) + (minor * 3) + (len(missing_sections) * 5)
    return min(score, 100)


def build_app(settings: Settings):
    violation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You review submissions against policy checklists and identify violations conservatively.",
            ),
            ("human", "Policy topic: {policy_topic}\nChecklist: {policy_text}\nSubmission: {submission}"),
        ]
    )
    missing_sections_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Determine which required sections are missing from the document. Return only the missing sections.",
            ),
            (
                "human",
                "Document type: {document_type}\nRequired sections: {required_sections}\nDocument text: {document_text}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    reviewer = violation_prompt | model.with_structured_output(ViolationAnalysis)
    missing_section_checker = missing_sections_prompt | model.with_structured_output(MissingSectionsResult)

    def parse_node(state: ComplianceState) -> ComplianceState:
        return {
            "policy_text": lookup_policy(state["policy_topic"]),
            "document_type": state["policy_topic"],
            "document_text": state["submission"],
        }

    def detect_violations_node(state: ComplianceState) -> ComplianceState:
        analysis = reviewer.invoke(
            {
                "policy_topic": state["policy_topic"],
                "policy_text": state["policy_text"],
                "submission": state["document_text"],
            }
        )
        return {
            "summary": analysis.summary,
            "violations": [violation.model_dump() for violation in analysis.violations],
            "escalation_required": analysis.escalation_required,
        }

    def missing_section_checker_node(state: ComplianceState) -> ComplianceState:
        required_sections = required_sections_for(state["document_type"])
        if not required_sections:
            return {"missing_sections": []}
        result = missing_section_checker.invoke(
            {
                "document_type": state["document_type"],
                "required_sections": ", ".join(required_sections),
                "document_text": state["document_text"],
            }
        )
        return {"missing_sections": result.missing_sections}

    def risk_scorer_node(state: ComplianceState) -> ComplianceState:
        score = calculate_risk_score(state.get("violations", []), state.get("missing_sections", []))
        return {
            "risk_score": score,
            "escalation_required": state.get("escalation_required", False) or score >= 60,
        }

    def recommendation_generator_node(state: ComplianceState) -> ComplianceState:
        recommendations = [
            f"Resolve {violation['severity']} issue: {violation['description']}"
            for violation in state.get("violations", [])
        ]
        recommendations.extend(
            f"Add a section covering: {section}" for section in state.get("missing_sections", [])
        )
        return {"recommendations": recommendations}

    def write_report_node(state: ComplianceState) -> ComplianceState:
        report = ComplianceReport(
            summary=state["summary"],
            risks=[violation["description"] for violation in state.get("violations", [])],
            missing_information=state.get("missing_sections", []),
            recommendations=state.get("recommendations", []),
            risk_score=state["risk_score"],
            escalation_required=state.get("escalation_required", False),
        )
        return {"report": report.model_dump()}

    graph = StateGraph(ComplianceState)
    graph.add_node("parse", parse_node)
    graph.add_node("detect_violations", detect_violations_node)
    graph.add_node("check_missing_sections", missing_section_checker_node)
    graph.add_node("score_risk", risk_scorer_node)
    graph.add_node("generate_recommendations", recommendation_generator_node)
    graph.add_node("write_report", write_report_node)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "detect_violations")
    graph.add_edge("detect_violations", "check_missing_sections")
    graph.add_edge("check_missing_sections", "score_risk")
    graph.add_edge("score_risk", "generate_recommendations")
    graph.add_edge("generate_recommendations", "write_report")
    graph.add_edge("write_report", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
