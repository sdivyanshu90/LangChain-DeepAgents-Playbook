import pytest

from langchain_core.documents import Document

from research_digest_builder.config import Settings
from research_digest_builder.digest import brief_to_document, build_digest, select_relevant_briefs
from research_digest_builder.renderer import render_markdown
from research_digest_builder.schemas import DigestReport, SourceBrief


def test_render_markdown_includes_digest_sections() -> None:
    digest = DigestReport(
        topic="AI Support Automation",
        audience="Operations leaders",
        executive_summary="The sources show growing interest in automation with clear governance needs.",
        themes=["Quality control matters", "Escalation paths remain essential"],
        recommended_actions=["Pilot in a narrow workflow"],
        open_questions=["How will accuracy be measured?"],
    )
    briefs = [
        SourceBrief(
            source="reports/report-1.md",
            title="Report 1",
            summary="A brief summary.",
            key_findings=["Finding 1"],
            risks=["Risk 1"],
        )
    ]

    rendered = render_markdown(digest, briefs)

    assert "## Executive Summary" in rendered
    assert "## Source Briefs" in rendered
    assert "### Report 1" in rendered


def test_brief_to_document_preserves_source_metadata() -> None:
    brief = SourceBrief(
        source="reports/report-1.md",
        title="Report 1",
        summary="A brief summary.",
        key_findings=["Finding 1"],
        risks=["Risk 1"],
    )

    document = brief_to_document(brief)

    assert document.metadata["source"] == "reports/report-1.md"
    assert "Finding 1" in document.page_content


def test_select_relevant_briefs_uses_retrieval_results(monkeypatch) -> None:
    briefs = [
        SourceBrief(source="a.md", title="A", summary="Alpha summary.", key_findings=[], risks=[]),
        SourceBrief(source="b.md", title="B", summary="Beta summary.", key_findings=[], risks=[]),
    ]

    class FakeIndex:
        def similarity_search(self, topic: str, k: int):
            assert topic == "beta topic"
            assert k == 2
            return [Document(page_content="Beta summary.", metadata={"source": "b.md"})]

    monkeypatch.setattr("research_digest_builder.digest.build_brief_index", lambda briefs, settings: FakeIndex())

    selected = select_relevant_briefs(
        briefs,
        "beta topic",
        Settings(model="gpt-4.1-mini", embedding_model="text-embedding-3-small", temperature=0.0),
    )

    assert [brief.source for brief in selected] == ["b.md"]


def test_build_digest_rejects_empty_briefs() -> None:
    with pytest.raises(ValueError, match="At least one source brief is required"):
        build_digest(
            briefs=[],
            topic="AI automation",
            audience="operators",
            settings=Settings(
                model="gpt-4.1-mini",
                embedding_model="text-embedding-3-small",
                temperature=0.0,
            ),
        )
