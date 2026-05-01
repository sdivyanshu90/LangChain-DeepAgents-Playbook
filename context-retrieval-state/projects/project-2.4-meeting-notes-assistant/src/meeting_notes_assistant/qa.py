from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import DecisionAnswer


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\nContent: {document.page_content}"
        for document in documents
    )


def answer_question(*, question: str, retriever, settings: Settings) -> DecisionAnswer:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieved_docs = retriever.invoke(question)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions about prior meetings using only the provided summary context.",
            ),
            (
                "human",
                "Question: {question}\n\nMeeting summary context:\n{context}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = prompt | model.with_structured_output(DecisionAnswer)
    return chain.invoke({"question": question.strip(), "context": format_context(retrieved_docs)})
