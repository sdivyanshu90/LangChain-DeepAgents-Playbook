from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_settings
from .indexer import build_retriever
from .loaders import load_documents
from .qa import answer_question
from .renderer import render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask grounded questions against a document directory.")
    parser.add_argument("--docs-dir", required=True, help="Directory containing source documents.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--session", default="default", help="Session identifier for follow-up questions.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output-file", help="Optional path to save the output.")
    return parser.parse_args()


def write_output(content: str, output_file: str | None) -> None:
    if output_file:
        Path(output_file).write_text(content, encoding="utf-8")
        return
    print(content, end="")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-2.2-knowledge-base-qa")
    args = parse_args()
    settings = load_settings()
    documents = load_documents(args.docs_dir)
    retriever = build_retriever(documents, settings)
    answer_bundle = answer_question(
        question=args.question,
        retriever=retriever,
        settings=settings,
        session_id=args.session,
    )

    if args.format == "markdown":
        rendered = render_markdown(args.question, answer_bundle)
    else:
        rendered = answer_bundle.model_dump_json(indent=2) + "\n"

    write_output(rendered, args.output_file)


if __name__ == "__main__":
    main()
