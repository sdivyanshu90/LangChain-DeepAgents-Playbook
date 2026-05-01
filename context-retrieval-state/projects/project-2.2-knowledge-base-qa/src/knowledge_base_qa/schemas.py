from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str = Field(..., description="Relative path of the cited source document.")
    excerpt: str = Field(..., description="Short excerpt supporting the answer.")


class AnswerBundle(BaseModel):
    answer: str = Field(..., description="Grounded answer to the user's question.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Evidence references supporting the answer.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Important uncertainties or missing information.",
    )
