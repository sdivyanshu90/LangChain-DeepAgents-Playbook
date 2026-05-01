import pytest

from langchain_core.documents import Document

from knowledge_base_qa.config import Settings
from knowledge_base_qa.qa import SESSION_HISTORIES, answer_question, format_context, get_session_history
from knowledge_base_qa.renderer import render_markdown
from knowledge_base_qa.schemas import AnswerBundle, Citation


def test_render_markdown_includes_sections() -> None:
    answer = AnswerBundle(
        answer="Launch readiness requires finance sign-off.",
        citations=[
            Citation(
                source_id="runbooks/billing.md",
                excerpt="Launch readiness review happens after quota approval is confirmed.",
            )
        ],
        gaps=["The timeline for final sign-off is not specified."],
    )

    rendered = render_markdown("What blocks launch?", answer)

    assert "# Question" in rendered
    assert "# Citations" in rendered
    assert "# Gaps" in rendered


def test_format_context_includes_source_ids() -> None:
    documents = [Document(page_content="Launch needs finance sign-off.", metadata={"source_id": "finance/runbook.md"})]

    rendered = format_context(documents)

    assert "Source ID: finance/runbook.md" in rendered


def test_get_session_history_reuses_same_session() -> None:
    SESSION_HISTORIES.clear()

    first = get_session_history("demo")
    second = get_session_history("demo")

    assert first is second


def test_answer_question_rejects_empty_question() -> None:
    class DummyRetriever:
        def invoke(self, _question: str):
            raise AssertionError("Retriever should not be called for blank questions.")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(
            question="   ",
            retriever=DummyRetriever(),
            settings=Settings(
                model="gpt-4.1-mini",
                embedding_model="text-embedding-3-small",
                temperature=0.0,
                chunk_size=800,
                chunk_overlap=100,
            ),
        )
