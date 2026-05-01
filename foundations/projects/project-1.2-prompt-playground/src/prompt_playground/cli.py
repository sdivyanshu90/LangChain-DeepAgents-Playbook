from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .chain import OUTPUT_MODES, run_prompt
from .config import load_settings
from .presets import PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment with prompt presets and output formats.")
    parser.add_argument("--preset", default="teacher", choices=sorted(PRESETS))
    parser.add_argument("--task", help="Task or question to send to the model.")
    parser.add_argument("--audience", default="software developers")
    parser.add_argument("--tone", default="clear and practical")
    parser.add_argument("--format", dest="output_mode", default="text", choices=OUTPUT_MODES)
    parser.add_argument("--output-file", help="Optional path to save the result.")
    parser.add_argument("--list-presets", action="store_true", help="Print the available presets.")
    return parser.parse_args()


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

    configure_tracing("project-1.2-prompt-playground")
    args = parse_args()

    if args.list_presets:
        write_output("\n".join(sorted(PRESETS)), args.output_file)
        return

    if not args.task:
        raise SystemExit("Provide --task unless you are using --list-presets.")

    settings = load_settings()
    result = run_prompt(
        task=args.task,
        audience=args.audience,
        tone=args.tone,
        preset_name=args.preset,
        output_mode=args.output_mode,
        settings=settings,
    )

    if hasattr(result, "model_dump_json"):
        rendered = result.model_dump_json(indent=2)
    else:
        rendered = str(result)

    write_output(rendered, args.output_file)


if __name__ == "__main__":
    main()
