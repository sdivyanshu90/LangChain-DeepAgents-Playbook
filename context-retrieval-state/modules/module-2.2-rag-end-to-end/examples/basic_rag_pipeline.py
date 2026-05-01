from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_docs(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{document.metadata['source']}] {document.page_content}" for document in documents
    )


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    source_docs = [
        Document(
            page_content=(
                "The billing migration launch depends on finance approving the quota model. "
                "Launch readiness review happens after quota approval is confirmed."
            ),
            metadata={"source": "billing_runbook.md"},
        ),
        Document(
            page_content=(
                "Customer success must receive the final migration FAQ before launch week. "
                "Training is scheduled two business days before release."
            ),
            metadata={"source": "enablement_plan.md"},
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=140, chunk_overlap=30)
    chunks = splitter.split_documents(source_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the provided context. If the context is incomplete, say so clearly.",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}",
            ),
        ]
    )
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    chain = prompt | model | StrOutputParser()

    question = "What must happen before the billing migration can launch?"
    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke({"question": question, "context": format_docs(retrieved_docs)})

    print(answer)


if __name__ == "__main__":
    main()
