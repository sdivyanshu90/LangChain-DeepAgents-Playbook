from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_note_documents(notes_dir: str) -> list[Document]:
    root = Path(notes_dir)
    if not root.exists():
        raise FileNotFoundError(f"Notes directory not found: {notes_dir}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            relative_path = path.relative_to(root).as_posix()
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": relative_path, "title": path.stem},
                )
            )

    if not documents:
        raise ValueError("No supported meeting note files were found.")

    return documents
