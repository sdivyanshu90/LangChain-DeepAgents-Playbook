from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_retriever(documents: list[Document], *, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_size"] = chunk_size

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return store.as_retriever(search_kwargs={"k": 3})


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    documents = [
        Document(
            page_content=(
                "The expense policy requires receipts for any reimbursement above fifty dollars. "
                "Managers must approve conference travel before booking. International travel also requires legal review."
            ),
            metadata={"source": "expense_policy.md", "department": "finance"},
        ),
        Document(
            page_content=(
                "Laptop replacements are handled by IT. Security keys must be rotated every ninety days. "
                "Employees must report phishing attempts within one business day."
            ),
            metadata={"source": "security_policy.md", "department": "security"},
        ),
    ]

    question = "What approvals are required before conference travel?"
    settings = [(90, 20), (180, 40)]

    for chunk_size, chunk_overlap in settings:
        retriever = build_retriever(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results = retriever.invoke(question)
        print(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

        for document in results:
            print(
                f"- source={document.metadata['source']} chunk_index={document.metadata['chunk_index']} "
                f"content={document.page_content[:90]}..."
            )

        print("---")


if __name__ == "__main__":
    main()
