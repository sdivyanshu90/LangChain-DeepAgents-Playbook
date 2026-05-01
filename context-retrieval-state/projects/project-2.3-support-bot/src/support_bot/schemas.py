from __future__ import annotations

from pydantic import BaseModel, Field


class SupportDecision(BaseModel):
    answer: str = Field(..., description="Grounded answer to the support question.")
    cited_policies: list[str] = Field(
        default_factory=list,
        description="Policy documents that support the answer.",
    )
    confidence: str = Field(..., description="One of high, medium, or low.")
    escalation_required: bool = Field(
        ..., description="Whether the question should be escalated to a human or specialist team."
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Why escalation is needed when the evidence is weak or incomplete.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Relevant facts that were not clearly available in the policy context.",
    )
