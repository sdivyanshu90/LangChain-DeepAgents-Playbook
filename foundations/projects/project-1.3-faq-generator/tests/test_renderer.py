import pytest

from faq_generator.chain import build_faq_document, generate_faq
from faq_generator.config import Settings
from faq_generator.renderer import render_markdown
from faq_generator.schemas import FAQDocument, FAQItem


def test_markdown_renderer_outputs_sections() -> None:
    document = FAQDocument(
        title="Billing FAQ",
        audience="Support team",
        approved=[
            FAQItem(
                question="When does the migration start?",
                answer="The migration starts next Monday.",
                category="timeline",
                confidence=0.92,
                tags=["migration", "billing"],
            )
        ],
    )

    rendered = render_markdown(document)

    assert "# Billing FAQ" in rendered
    assert "## When does the migration start?" in rendered
    assert "Category: timeline" in rendered
    assert "Tags: migration, billing" in rendered


def test_build_faq_document_splits_low_confidence_items() -> None:
    approved = FAQItem(
        question="What changed?",
        answer="The launch date moved.",
        category="release",
        confidence=0.9,
        tags=["launch"],
    )
    needs_review = FAQItem(
        question="Will pricing change?",
        answer="Pricing might change later.",
        category="pricing",
        confidence=0.4,
        tags=["pricing"],
    )

    document = build_faq_document(
        title="Release FAQ",
        audience="Customers",
        items=[approved, needs_review],
    )

    assert len(document.approved) == 1
    assert len(document.needs_review) == 1
    assert document.needs_review[0].question == "Will pricing change?"


def test_generate_faq_rejects_empty_source_text() -> None:
    with pytest.raises(ValueError, match="Source text cannot be empty"):
        generate_faq(
            source_text="   ",
            title="Empty FAQ",
            audience="Operators",
            settings=Settings(model="gpt-4.1-mini", temperature=0.0),
        )
