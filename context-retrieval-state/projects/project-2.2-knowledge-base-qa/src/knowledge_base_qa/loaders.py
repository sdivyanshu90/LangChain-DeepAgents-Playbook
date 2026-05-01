from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}


def load_documents(docs_dir: str) -> list[Document]:
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            relative_path = path.relative_to(root).as_posix()
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source_id": relative_path, "source": relative_path},
                )
            )

    if not documents:
        raise ValueError("No supported documents were found in the docs directory.")

    return documents
