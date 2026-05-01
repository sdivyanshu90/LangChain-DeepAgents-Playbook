from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import MeetingSummary


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You convert meeting notes into grounded structured summaries. Only include decisions and actions that are supported by the note.",
        ),
        (
            "human",
            "Source: {source}\n"
            "Title hint: {title}\n\n"
            "Meeting notes:\n{note_text}",
        ),
    ]
)


def build_summary_chain(settings: Settings):
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    return PROMPT | model.with_structured_output(MeetingSummary)


def summarize_document(document: Document, chain) -> MeetingSummary:
    return chain.invoke(
        {
            "source": document.metadata["source"],
            "title": document.metadata["title"],
            "note_text": document.page_content,
        }
    )


def summarize_documents(documents: list[Document], settings: Settings, *, chain=None) -> list[MeetingSummary]:
    summary_chain = chain or build_summary_chain(settings)

    summaries: list[MeetingSummary] = []
    for document in documents:
        summaries.append(summarize_document(document, summary_chain))

    return summaries
