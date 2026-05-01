from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_settings
from .digest import build_digest
from .loaders import load_documents
from .renderer import render_markdown
from .summarizer import summarize_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research digest from source documents.")
    parser.add_argument("--docs-dir", required=True, help="Directory containing source documents.")
    parser.add_argument("--topic", required=True, help="Topic for the final digest.")
    parser.add_argument("--audience", required=True, help="Audience for the final digest.")
    parser.add_argument("--json-output", help="Optional file path for JSON output.")
    parser.add_argument("--markdown-output", help="Optional file path for Markdown output.")
    return parser.parse_args()


def maybe_write(path: str | None, content: str) -> None:
    if path:
        Path(path).write_text(content, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-2.5-research-digest-builder")
    args = parse_args()
    settings = load_settings()
    documents = load_documents(args.docs_dir)
    briefs = summarize_documents(documents, settings)
    digest = build_digest(briefs=briefs, topic=args.topic, audience=args.audience, settings=settings)

    json_payload = {
        "digest": digest.model_dump(),
        "source_briefs": [brief.model_dump() for brief in briefs],
    }
    markdown_payload = render_markdown(digest, briefs)

    maybe_write(args.json_output, json.dumps(json_payload, indent=2) + "\n")
    maybe_write(args.markdown_output, markdown_payload)

    if not args.markdown_output:
        print(markdown_payload, end="")


if __name__ == "__main__":
    main()
