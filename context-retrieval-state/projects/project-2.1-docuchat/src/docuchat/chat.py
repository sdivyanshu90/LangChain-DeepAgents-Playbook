from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .config import Settings
from .session import FileChatSessionStore


def format_context(documents: list[Document]) -> str:
    parts = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "n/a")
        parts.append(f"[{source} p.{page}] {document.page_content}")
    return "\n\n".join(parts)


def format_citations(documents: list[Document]) -> list[str]:
    citations: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "n/a")
        citation = f"{source} p.{page}"
        if citation not in citations:
            citations.append(citation)
    return citations


def answer_question(
    *,
    question: str,
    retriever,
    settings: Settings,
    session_store: FileChatSessionStore,
    session_id: str,
) -> tuple[str, list[str]]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    history = session_store.load_messages(session_id, limit=settings.history_window * 2)
    retrieved_docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions about the provided PDF. Use the retrieved context and keep the answer grounded. "
                "If the context is incomplete, say that clearly.",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "Question: {question}\n\nRetrieved PDF context:\n{context}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke(
        {
            "history": history,
            "question": question.strip(),
            "context": format_context(retrieved_docs),
        }
    )

    session_store.append_turn(session_id, question.strip(), answer)
    return answer, format_citations(retrieved_docs)
