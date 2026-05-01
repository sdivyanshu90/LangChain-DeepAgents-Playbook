from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from .config import load_settings
from .workflow import build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the workflow recovery agent.")
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-3.12-workflow-recovery-agent")
    args = parse_args()
    app = build_app(load_settings())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    settings = load_settings()
    result = app.invoke(
        {
            "task": args.task,
            "retries": 0,
            "max_retries": settings.max_retries,
            "checkpoints": [],
            "partial_results": [],
            "fallback_used": False,
        },
        config=config,
    )
    print(result["result"])
    print("\nCheckpoints:")
    for item in result.get("checkpoints", []):
        print(f"- {item}")


if __name__ == "__main__":
    main()
