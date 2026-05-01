from __future__ import annotations

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import ClassifiedFAQItems, ExtractedFAQPairs, FAQDocument, FAQItem


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract grounded FAQ candidate pairs from rough operational or product notes. "
            "Do not invent product capabilities or policies that are not present in the source text.",
        ),
        (
            "human",
            "Maximum FAQ items: {max_faqs}\n\n"
            "Source notes:\n{source_text}",
        ),
    ]
)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify extracted FAQ entries, assign a category, and provide a confidence score between 0.0 and 1.0. "
            "Keep answers grounded in the supplied candidate pairs.",
        ),
        (
            "human",
            "Title: {title}\n"
            "Audience: {audience}\n"
            "Candidate FAQ pairs:\n{candidate_pairs}",
        ),
    ]
)


def build_faq_document(*, title: str, audience: str, items: list[FAQItem]) -> FAQDocument:
    approved = [item for item in items if item.confidence >= 0.7]
    needs_review = [item for item in items if item.confidence < 0.7]
    return FAQDocument(title=title, audience=audience, approved=approved, needs_review=needs_review)


def build_generator(settings: Settings):
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    extraction_chain = EXTRACTION_PROMPT | model.with_structured_output(ExtractedFAQPairs)
    classification_chain = CLASSIFICATION_PROMPT | model.with_structured_output(ClassifiedFAQItems)

    def build_classification_payload(payload: dict) -> dict:
        extraction = payload["extraction"]
        candidate_pairs = extraction.pairs[: payload["max_faqs"]]
        return {
            "title": payload["title"],
            "audience": payload["audience"],
            "candidate_pairs": "\n\n".join(candidate_pairs),
        }

    def build_document(payload: dict) -> FAQDocument:
        return build_faq_document(
            title=payload["title"],
            audience=payload["audience"],
            items=payload["classified"].items,
        )

    return (
        RunnablePassthrough.assign(extraction=extraction_chain)
        | RunnablePassthrough.assign(
            classified=RunnableLambda(build_classification_payload) | classification_chain
        )
        | RunnableLambda(build_document)
    )


def generate_faq(
    *,
    source_text: str,
    title: str,
    audience: str,
    settings: Settings,
    max_faqs: int = 6,
) -> FAQDocument:
    cleaned_text = source_text.strip()
    if not cleaned_text:
        raise ValueError("Source text cannot be empty.")

    generator = build_generator(settings)
    return generator.invoke(
        {
            "source_text": cleaned_text,
            "title": title.strip(),
            "audience": audience.strip(),
            "max_faqs": max_faqs,
        }
    )
