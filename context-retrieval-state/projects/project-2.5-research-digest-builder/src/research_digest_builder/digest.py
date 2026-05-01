from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import DigestReport, SourceBrief


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You synthesize multiple source briefs into a concise research digest for the requested audience.",
        ),
        (
            "human",
            "Topic: {topic}\n"
            "Audience: {audience}\n\n"
            "Source briefs:\n{briefs_json}",
        ),
    ]
)


def brief_to_document(brief: SourceBrief) -> Document:
    content = [brief.summary]
    if brief.key_findings:
        content.append("Key findings: " + "; ".join(brief.key_findings))
    if brief.risks:
        content.append("Risks: " + "; ".join(brief.risks))
    return Document(
        page_content="\n".join(content),
        metadata={"source": brief.source, "title": brief.title},
    )


def build_brief_index(briefs: list[SourceBrief], settings: Settings) -> FAISS:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    documents = [brief_to_document(brief) for brief in briefs]
    return FAISS.from_documents(documents, embeddings)


def select_relevant_briefs(
    briefs: list[SourceBrief],
    topic: str,
    settings: Settings,
    *,
    k: int = 5,
) -> list[SourceBrief]:
    index = build_brief_index(briefs, settings)
    results = index.similarity_search(topic, k=min(k, len(briefs)))
    brief_by_source = {brief.source: brief for brief in briefs}

    selected: list[SourceBrief] = []
    for document in results:
        source = document.metadata.get("source")
        brief = brief_by_source.get(source)
        if brief and brief not in selected:
            selected.append(brief)
    return selected or briefs


def build_digest(
    *,
    briefs: list[SourceBrief],
    topic: str,
    audience: str,
    settings: Settings,
) -> DigestReport:
    if not briefs:
        raise ValueError("At least one source brief is required to build a digest.")

    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = PROMPT | model.with_structured_output(DigestReport)
    retrieved_briefs = select_relevant_briefs(briefs, topic, settings)
    briefs_json = json.dumps([brief.model_dump() for brief in retrieved_briefs], indent=2)
    return chain.invoke({"topic": topic, "audience": audience, "briefs_json": briefs_json})
