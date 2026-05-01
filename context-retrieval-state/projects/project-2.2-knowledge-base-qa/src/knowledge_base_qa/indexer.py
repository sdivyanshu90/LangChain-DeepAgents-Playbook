from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


def build_retriever(documents: list[Document], settings: Settings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})
