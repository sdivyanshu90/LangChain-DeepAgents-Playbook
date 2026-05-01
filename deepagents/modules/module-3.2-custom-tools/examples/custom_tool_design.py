from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class EscalationDecision(BaseModel):
    severity: Literal["low", "medium", "high"] = Field(..., description="Operational severity tier.")
    escalate: bool = Field(..., description="Whether a human escalation is required.")
    next_step: str = Field(..., description="Recommended next operational step.")


@tool(args_schema=EscalationDecision)
def build_escalation_note(severity: str, escalate: bool, next_step: str) -> str:
    """Format a deterministic escalation note for operations teams."""
    status = "ESCALATE" if escalate else "HANDLE IN WORKFLOW"
    return f"[{status}] severity={severity}; next_step={next_step}"


def main() -> None:
    note = build_escalation_note.invoke(
        {"severity": "high", "escalate": True, "next_step": "Page the incident commander immediately."}
    )
    print(note)


if __name__ == "__main__":
    main()
