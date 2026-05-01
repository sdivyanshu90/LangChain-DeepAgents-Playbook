from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}


def infer_department(path: Path, root: Path) -> str:
    relative_parts = path.relative_to(root).parts
    if len(relative_parts) > 1:
        return relative_parts[0]
    return "general"


def load_policy_documents(policies_dir: str) -> list[Document]:
    root = Path(policies_dir)
    if not root.exists():
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            relative_path = path.relative_to(root).as_posix()
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={
                        "source": relative_path,
                        "department": infer_department(path, root),
                    },
                )
            )

    if not documents:
        raise ValueError("No supported policy documents were found.")

    return documents
