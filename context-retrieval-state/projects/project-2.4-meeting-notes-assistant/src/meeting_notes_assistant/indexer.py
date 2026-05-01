from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from .config import Settings
from .schemas import MeetingSummary


def summary_to_document(summary: MeetingSummary) -> Document:
    action_lines = [
        f"owner={item.owner}; task={item.task}; deadline={item.deadline or 'none'}"
        for item in summary.action_items
    ]

    page_content = "\n".join(
        [
            f"Meeting title: {summary.meeting_title}",
            f"Summary: {summary.summary}",
            f"Decisions: {' | '.join(summary.decisions)}",
            f"Action items: {' | '.join(action_lines)}",
            f"Open questions: {' | '.join(summary.open_questions)}",
        ]
    )
    return Document(page_content=page_content, metadata={"source": summary.source})


def build_retriever(summaries: list[MeetingSummary], settings: Settings):
    documents = [summary_to_document(summary) for summary in summaries]
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})
