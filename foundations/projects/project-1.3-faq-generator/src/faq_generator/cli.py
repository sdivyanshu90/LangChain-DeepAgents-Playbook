from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chain import generate_faq
from .config import load_settings
from .renderer import render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FAQs from raw notes.")
    parser.add_argument("--source-file", help="Path to the raw notes file.")
    parser.add_argument("--title", required=True, help="Title for the FAQ document.")
    parser.add_argument("--audience", required=True, help="Audience the FAQ is written for.")
    parser.add_argument("--max-faqs", type=int, default=6, help="Maximum number of FAQ items.")
    parser.add_argument("--json-output", help="Optional file path for JSON output.")
    parser.add_argument("--markdown-output", help="Optional file path for Markdown output.")
    return parser.parse_args()


def read_source_text(source_file: str | None) -> str:
    if source_file:
        return Path(source_file).read_text(encoding="utf-8")

    if not sys.stdin.isatty():
        return sys.stdin.read()

    raise ValueError("Provide --source-file or pipe the source notes into standard input.")


def maybe_write(path: str | None, content: str) -> None:
    if path:
        Path(path).write_text(content, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-1.3-faq-generator")
    args = parse_args()
    source_text = read_source_text(args.source_file)
    settings = load_settings()

    faq = generate_faq(
        source_text=source_text,
        title=args.title,
        audience=args.audience,
        settings=settings,
        max_faqs=args.max_faqs,
    )

    faq_json = faq.model_dump_json(indent=2)
    faq_markdown = render_markdown(faq)

    maybe_write(args.json_output, faq_json + "\n")
    maybe_write(args.markdown_output, faq_markdown)

    if not args.markdown_output:
        print(faq_markdown, end="")


if __name__ == "__main__":
    main()
