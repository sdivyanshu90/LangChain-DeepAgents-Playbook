from __future__ import annotations

from .schemas import FAQDocument


def render_markdown(document: FAQDocument) -> str:
    lines = [f"# {document.title}", "", f"Audience: {document.audience}", ""]

    for item in document.approved:
        lines.append(f"## {item.question}")
        lines.append("")
        lines.append(item.answer)
        lines.append("")
        lines.append(f"Category: {item.category}")
        lines.append(f"Confidence: {item.confidence:.2f}")

        if item.tags:
            lines.append("")
            lines.append(f"Tags: {', '.join(item.tags)}")

        lines.append("")

    if document.needs_review:
        lines.extend(["## Needs Review", ""])
        for item in document.needs_review:
            lines.append(f"- {item.question} ({item.confidence:.2f})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
