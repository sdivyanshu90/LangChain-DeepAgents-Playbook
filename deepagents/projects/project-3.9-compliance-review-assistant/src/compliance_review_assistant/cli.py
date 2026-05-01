from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

from .config import load_settings
from .workflow import build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compliance review assistant.")
    parser.add_argument("--policy-topic", required=True)
    parser.add_argument("--submission", required=True)
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-3.9-compliance-review-assistant")
    args = parse_args()
    app = build_app(load_settings())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke({"policy_topic": args.policy_topic, "submission": args.submission}, config=config)
    print(json.dumps(result["report"], indent=2))


if __name__ == "__main__":
    main()
