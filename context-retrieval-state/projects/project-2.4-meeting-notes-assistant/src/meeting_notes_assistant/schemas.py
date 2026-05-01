from __future__ import annotations

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    owner: str = Field(..., description="Person or team responsible for the action item.")
    task: str = Field(..., description="Action item description.")
    deadline: str | None = Field(
        default=None,
        description="Deadline if the note clearly includes one.",
    )


class MeetingSummary(BaseModel):
    source: str = Field(..., description="Source file for the meeting note.")
    meeting_title: str = Field(..., description="Human-readable title for the meeting.")
    summary: str = Field(..., description="Short summary of the meeting.")
    decisions: list[str] = Field(
        default_factory=list,
        description="Concrete decisions captured in the note.",
    )
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Action items extracted from the note.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Open questions or unresolved issues.",
    )


class DecisionAnswer(BaseModel):
    answer: str = Field(..., description="Grounded answer derived from retrieved meeting summaries.")
    citations: list[str] = Field(
        default_factory=list,
        description="Meeting summary sources that support the answer.",
    )
