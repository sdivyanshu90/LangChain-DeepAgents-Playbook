from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import SupportDecision


def format_context(documents: list[Document]) -> str:
    blocks = []
    for document in documents:
        blocks.append(
            f"Source: {document.metadata.get('source', 'unknown')}\n"
            f"Department: {document.metadata.get('department', 'general')}\n"
            f"Content: {document.page_content}"
        )
    return "\n\n".join(blocks)


def retrieve_policy_documents(question: str, retriever, department: str | None = None) -> list[Document]:
    retrieved_docs = retriever.invoke(question)
    if department:
        filtered_docs = [
            document
            for document in retrieved_docs
            if document.metadata.get("department") in {department, "general"}
        ]
        if filtered_docs:
            return filtered_docs
    return retrieved_docs


def build_decision_chain(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an internal support policy assistant. Answer only from the provided policy context. "
                "If the evidence is incomplete, set escalation_required to true and explain why.",
            ),
            (
                "human",
                "Department scope: {department}\n"
                "Question: {question}\n\n"
                "Policy context:\n{context}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    return prompt | model.with_structured_output(SupportDecision)


def build_streaming_chain(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an internal support policy assistant. Answer only from the provided policy context. "
                "If the evidence is incomplete, say that clearly and explain what should be escalated.",
            ),
            (
                "human",
                "Department scope: {department}\n"
                "Question: {question}\n\n"
                "Policy context:\n{context}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    return prompt | model | StrOutputParser()


def answer_policy_question(
    *,
    question: str,
    retriever,
    settings: Settings,
    department: str | None = None,
) -> SupportDecision:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieved_docs = retrieve_policy_documents(question, retriever, department)
    chain = build_decision_chain(settings)
    return chain.invoke(
        {
            "department": department or "all",
            "question": question.strip(),
            "context": format_context(retrieved_docs),
        }
    )


def stream_policy_answer(
    *,
    question: str,
    retriever,
    settings: Settings,
    department: str | None = None,
):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieved_docs = retrieve_policy_documents(question, retriever, department)
    chain = build_streaming_chain(settings)
    return chain.stream(
        {
            "department": department or "all",
            "question": question.strip(),
            "context": format_context(retrieved_docs),
        }
    )


def write_streamlit_answer(
    *,
    question: str,
    retriever,
    settings: Settings,
    department: str | None = None,
):
    import streamlit as st

    with st.chat_message("assistant"):
        return st.write_stream(
            chunk for chunk in stream_policy_answer(
                question=question,
                retriever=retriever,
                settings=settings,
                department=department,
            )
        )
