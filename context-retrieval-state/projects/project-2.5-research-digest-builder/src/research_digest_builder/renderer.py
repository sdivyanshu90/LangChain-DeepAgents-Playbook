from __future__ import annotations

from .schemas import DigestReport, SourceBrief


def render_markdown(digest: DigestReport, briefs: list[SourceBrief]) -> str:
    lines = [
        f"# {digest.topic}",
        "",
        f"Audience: {digest.audience}",
        "",
        "## Executive Summary",
        "",
        digest.executive_summary,
        "",
    ]

    if digest.themes:
        lines.extend(["## Themes", ""])
        for theme in digest.themes:
            lines.append(f"- {theme}")
        lines.append("")

    if digest.recommended_actions:
        lines.extend(["## Recommended Actions", ""])
        for action in digest.recommended_actions:
            lines.append(f"- {action}")
        lines.append("")

    if digest.open_questions:
        lines.extend(["## Open Questions", ""])
        for question in digest.open_questions:
            lines.append(f"- {question}")
        lines.append("")

    lines.extend(["## Source Briefs", ""])
    for brief in briefs:
        lines.extend([f"### {brief.title}", "", f"Source: {brief.source}", "", brief.summary, ""])

    return "\n".join(lines).rstrip() + "\n"
