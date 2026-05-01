from __future__ import annotations

from pydantic import BaseModel, Field


class OutlineResponse(BaseModel):
    title: str = Field(..., description="Short title for the response.")
    audience: str = Field(..., description="Audience the answer is tailored for.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Primary points the response wants the reader to understand.",
    )
    recommended_next_step: str = Field(
        ..., description="One practical next step for the audience to take."
    )
