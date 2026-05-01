from __future__ import annotations

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


PERSIST_DIR = ".chroma_docuchat"


def build_retriever(documents: list[Document], settings: Settings, *, persist_directory: str = PERSIST_DIR):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = Chroma(
        collection_name="docuchat",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    if vector_store._collection.count() == 0 and chunks:
        vector_store.add_documents(chunks)
    return vector_store.as_retriever(search_kwargs={"k": 4})
