from __future__ import annotations

from pydantic import BaseModel, Field


class FAQItem(BaseModel):
    question: str = Field(..., description="A concise question a real user might ask.")
    answer: str = Field(..., description="A clear answer grounded in the source notes.")
    category: str = Field(..., description="The topical bucket assigned during classification.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the FAQ entry.")
    tags: list[str] = Field(
        default_factory=list,
        description="Short labels that help categorize the FAQ item.",
    )


class ExtractedFAQPairs(BaseModel):
    pairs: list[str] = Field(
        default_factory=list,
        description="Raw Q&A candidate pairs extracted from the source notes.",
    )


class ClassifiedFAQItems(BaseModel):
    items: list[FAQItem] = Field(
        default_factory=list,
        description="FAQ items after category and confidence classification.",
    )


class FAQDocument(BaseModel):
    title: str = Field(..., description="Title of the FAQ document.")
    audience: str = Field(..., description="Audience the FAQ is written for.")
    approved: list[FAQItem] = Field(
        default_factory=list,
        description="FAQ entries that cleared the confidence threshold.",
    )
    needs_review: list[FAQItem] = Field(
        default_factory=list,
        description="FAQ entries that require human review before publication.",
    )
