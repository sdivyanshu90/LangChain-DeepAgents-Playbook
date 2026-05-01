from __future__ import annotations

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    owner: str = Field(..., description="Person or team responsible for the task.")
    task: str = Field(..., description="Concrete task that needs to happen next.")
    deadline: str | None = Field(
        default=None,
        description="Deadline if the source text clearly contains one; otherwise null.",
    )


class StructuredNote(BaseModel):
    topic: str = Field(..., description="Primary topic or subject of the note.")
    summary: str = Field(..., description="Short summary of the note in plain English.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Important facts that are explicitly supported by the source text.",
    )
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Concrete next steps that appear in the note.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Concerns, blockers, or unresolved issues mentioned in the note.",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Questions that remain unanswered after reading the note.",
    )
