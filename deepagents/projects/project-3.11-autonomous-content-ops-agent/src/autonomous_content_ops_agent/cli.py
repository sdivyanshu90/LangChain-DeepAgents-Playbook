from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from .config import load_settings
from .workflow import build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous content ops agent.")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--audience", required=True)
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-3.11-autonomous-content-ops-agent")
    args = parse_args()
    app = build_app(load_settings())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke({"topic": args.topic, "audience": args.audience}, config=config)
    print(result["revised_draft"])


if __name__ == "__main__":
    main()
