from __future__ import annotations

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory as InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import AnswerBundle


SESSION_HISTORIES: dict[str, InMemoryChatMessageHistory] = {}


def format_context(documents: list[Document]) -> str:
    blocks = []
    for document in documents:
        source_id = document.metadata.get("source_id", "unknown")
        blocks.append(f"Source ID: {source_id}\nContent: {document.page_content}")
    return "\n\n".join(blocks)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    history = SESSION_HISTORIES.get(session_id)
    if history is None:
        history = InMemoryChatMessageHistory()
        SESSION_HISTORIES[session_id] = history
    return history


def build_answer_chain(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using only the provided context. Include only citations that are clearly supported by the context. "
                "If the evidence is incomplete, record that in gaps.",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}",
            ),
        ]
    )

    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = prompt | model.with_structured_output(AnswerBundle)
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def answer_question(*, question: str, retriever, settings: Settings, session_id: str = "default") -> AnswerBundle:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieved_docs = retriever.invoke(question)
    chain = build_answer_chain(settings)
    return chain.invoke(
        {"question": question.strip(), "context": format_context(retrieved_docs)},
        config={"configurable": {"session_id": session_id}},
    )
