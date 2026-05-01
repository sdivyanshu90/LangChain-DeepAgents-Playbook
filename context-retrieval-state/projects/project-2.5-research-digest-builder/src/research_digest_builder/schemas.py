from __future__ import annotations

from pydantic import BaseModel, Field


class SourceBrief(BaseModel):
    source: str = Field(..., description="Source file identifier.")
    title: str = Field(..., description="Short title for the source.")
    summary: str = Field(..., description="Short grounded summary of the source.")
    key_findings: list[str] = Field(
        default_factory=list,
        description="Primary findings from the source.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Risks, limitations, or concerns raised by the source.",
    )


class DigestReport(BaseModel):
    topic: str = Field(..., description="Topic the digest is about.")
    audience: str = Field(..., description="Audience the digest is written for.")
    executive_summary: str = Field(..., description="Short synthesis across the source briefs.")
    themes: list[str] = Field(
        default_factory=list,
        description="Cross-document themes that appear repeatedly.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Suggested next actions based on the overall synthesis.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions that remain after synthesis.",
    )
