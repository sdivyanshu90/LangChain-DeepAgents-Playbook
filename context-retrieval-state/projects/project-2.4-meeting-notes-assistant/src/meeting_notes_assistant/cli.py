from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_settings
from .indexer import build_retriever
from .loaders import load_note_documents
from .qa import answer_question
from .renderer import render_markdown
from .summarizer import summarize_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize meeting notes and answer follow-up questions.")
    parser.add_argument("--notes-dir", required=True, help="Directory containing meeting notes.")
    parser.add_argument("--question", help="Optional question about the summarized meetings.")
    parser.add_argument("--json-output", help="Optional file path for structured summaries.")
    parser.add_argument("--markdown-output", help="Optional file path for Markdown summaries.")
    return parser.parse_args()


def maybe_write(path: str | None, content: str) -> None:
    if path:
        Path(path).write_text(content, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-2.4-meeting-notes-assistant")
    args = parse_args()
    settings = load_settings()
    documents = load_note_documents(args.notes_dir)
    summaries = summarize_documents(documents, settings)

    summaries_json = json.dumps([summary.model_dump() for summary in summaries], indent=2)
    summaries_markdown = render_markdown(summaries)

    maybe_write(args.json_output, summaries_json + "\n")
    maybe_write(args.markdown_output, summaries_markdown)

    if args.question:
        retriever = build_retriever(summaries, settings)
        answer = answer_question(question=args.question, retriever=retriever, settings=settings)
        print(answer.model_dump_json(indent=2))
        return

    if not args.markdown_output:
        print(summaries_markdown, end="")


if __name__ == "__main__":
    main()
