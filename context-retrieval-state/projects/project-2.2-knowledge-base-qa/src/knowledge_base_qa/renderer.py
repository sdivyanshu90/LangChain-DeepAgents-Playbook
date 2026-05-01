from __future__ import annotations

from .schemas import AnswerBundle


def render_markdown(question: str, answer_bundle: AnswerBundle) -> str:
    lines = [f"# Question", "", question, "", "# Answer", "", answer_bundle.answer, ""]

    if answer_bundle.citations:
        lines.extend(["# Citations", ""])
        for citation in answer_bundle.citations:
            lines.append(f"- {citation.source_id}: {citation.excerpt}")
        lines.append("")

    if answer_bundle.gaps:
        lines.extend(["# Gaps", ""])
        for gap in answer_bundle.gaps:
            lines.append(f"- {gap}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
