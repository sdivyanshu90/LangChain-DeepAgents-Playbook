from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from .chat import answer_question
from .config import load_settings
from .indexer import PERSIST_DIR, build_retriever
from .loaders import load_pdf_documents
from .session import FileChatSessionStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a PDF using retrieval and session history.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF document.")
    parser.add_argument("--question", required=True, help="Question to ask about the PDF.")
    parser.add_argument("--session-id", default="default-session", help="Conversation session identifier.")
    parser.add_argument(
        "--session-dir",
        default=".docuchat_sessions",
        help="Directory for persisted chat history.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the persisted Chroma index.")
    return parser.parse_args()


def reset_persist_dir(persist_directory: str = PERSIST_DIR) -> None:
    persist_path = Path(persist_directory)
    if persist_path.exists():
        shutil.rmtree(persist_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-2.1-docuchat")
    args = parse_args()
    settings = load_settings()
    documents = load_pdf_documents(args.pdf)
    if args.rebuild:
        reset_persist_dir()
    retriever = build_retriever(documents, settings)
    session_store = FileChatSessionStore(args.session_dir)

    answer, citations = answer_question(
        question=args.question,
        retriever=retriever,
        settings=settings,
        session_store=session_store,
        session_id=args.session_id,
    )

    print(answer)
    if citations:
        print("\nSources:")
        for citation in citations:
            print(f"- {citation}")


if __name__ == "__main__":
    main()
