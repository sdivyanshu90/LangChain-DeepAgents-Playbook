from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from .config import load_settings
from .workflow import build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data query agent.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--question", required=True)
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-3.8-data-query-agent")
    args = parse_args()
    app = build_app(load_settings())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke({"db_path": args.db_path, "question": args.question}, config=config)
    print(result.get("answer", "No answer generated."))
    print("\nSQL:")
    print(result.get("sql", "No SQL generated."))


if __name__ == "__main__":
    main()
