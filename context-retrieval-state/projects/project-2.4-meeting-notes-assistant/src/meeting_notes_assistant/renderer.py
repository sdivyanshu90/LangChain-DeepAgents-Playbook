from __future__ import annotations

from .schemas import MeetingSummary


def render_markdown(summaries: list[MeetingSummary]) -> str:
    lines = ["# Meeting Notes Digest", ""]

    for summary in summaries:
        lines.extend([f"## {summary.meeting_title}", "", f"Source: {summary.source}", "", summary.summary, ""])

        if summary.decisions:
            lines.append("### Decisions")
            lines.append("")
            for decision in summary.decisions:
                lines.append(f"- {decision}")
            lines.append("")

        if summary.action_items:
            lines.append("### Action Items")
            lines.append("")
            for item in summary.action_items:
                deadline = item.deadline or "No deadline"
                lines.append(f"- {item.owner}: {item.task} ({deadline})")
            lines.append("")

        if summary.open_questions:
            lines.append("### Open Questions")
            lines.append("")
            for question in summary.open_questions:
                lines.append(f"- {question}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
