from pathlib import Path

import pytest

from langchain_core.documents import Document

from support_bot.bot import answer_policy_question, format_context, retrieve_policy_documents
from support_bot.config import Settings
from support_bot.loaders import infer_department


def test_infer_department_from_nested_path(tmp_path: Path) -> None:
    policy_dir = tmp_path / "finance"
    policy_dir.mkdir()
    policy_file = policy_dir / "expense.md"
    policy_file.write_text("policy text", encoding="utf-8")

    assert infer_department(policy_file, tmp_path) == "finance"


def test_infer_department_defaults_to_general(tmp_path: Path) -> None:
    policy_file = tmp_path / "employee_handbook.md"
    policy_file.write_text("policy text", encoding="utf-8")

    assert infer_department(policy_file, tmp_path) == "general"


def test_format_context_includes_policy_metadata() -> None:
    documents = [
        Document(
            page_content="Travel requires manager approval.",
            metadata={"source": "finance/travel.md", "department": "finance"},
        )
    ]

    rendered = format_context(documents)

    assert "Source: finance/travel.md" in rendered
    assert "Department: finance" in rendered


def test_retrieve_policy_documents_applies_department_filter() -> None:
    class DummyRetriever:
        def invoke(self, _question: str):
            return [
                Document(page_content="Finance answer", metadata={"department": "finance"}),
                Document(page_content="Security answer", metadata={"department": "security"}),
            ]

    results = retrieve_policy_documents("approval?", DummyRetriever(), department="finance")

    assert len(results) == 1
    assert results[0].metadata["department"] == "finance"


def test_answer_policy_question_rejects_empty_question() -> None:
    class DummyRetriever:
        def invoke(self, _question: str):
            raise AssertionError("Retriever should not be called for blank questions.")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_policy_question(
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
