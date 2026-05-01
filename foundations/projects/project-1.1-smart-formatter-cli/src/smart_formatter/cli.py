from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chain import format_note, stream_format
from .config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert messy notes into strict JSON using LangChain structured outputs."
    )
    parser.add_argument("--input-file", help="Path to a text file containing the raw note.")
    parser.add_argument("--output-file", help="Optional path to write the JSON output.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON response.")
    parser.add_argument("--stream", action="store_true", help="Stream raw formatter output to standard output.")
    return parser.parse_args()


def read_input_text(input_file: str | None) -> str:
    if input_file:
        return Path(input_file).read_text(encoding="utf-8")

    if not sys.stdin.isatty():
        return sys.stdin.read()

    raise ValueError("Provide --input-file or pipe note text into standard input.")


def write_output(text: str, output_file: str | None) -> None:
    if output_file:
        Path(output_file).write_text(text + "\n", encoding="utf-8")
        return

    print(text)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-1.1-smart-formatter-cli")
    args = parse_args()
    raw_text = read_input_text(args.input_file)
    settings = load_settings()

    if args.stream:
        chunks = []
        for chunk in stream_format(raw_text=raw_text, settings=settings):
            text = str(chunk)
            chunks.append(text)
            print(text, end="", flush=True)
        if args.output_file:
            write_output("".join(chunks), args.output_file)
        elif not chunks or not chunks[-1].endswith("\n"):
            print()
        return

    result = format_note(raw_text=raw_text, settings=settings)
    rendered = result.model_dump_json(indent=2 if args.pretty else None)
    write_output(rendered, args.output_file)


if __name__ == "__main__":
    main()
