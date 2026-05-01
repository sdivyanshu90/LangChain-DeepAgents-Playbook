from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import SourceBrief


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You create grounded research source briefs. Keep the summary concise and do not invent findings.",
        ),
        (
            "human",
            "Source: {source}\n"
            "Title hint: {title}\n\n"
            "Document text:\n{text}",
        ),
    ]
)


def summarize_documents(documents: list[Document], settings: Settings) -> list[SourceBrief]:
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = PROMPT | model.with_structured_output(SourceBrief)

    briefs: list[SourceBrief] = []
    for document in documents:
        briefs.append(
            chain.invoke(
                {
                    "source": document.metadata["source"],
                    "title": document.metadata["title"],
                    "text": document.page_content,
                }
            )
        )

    return briefs
