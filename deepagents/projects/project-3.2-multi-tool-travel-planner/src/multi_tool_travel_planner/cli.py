from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from .config import load_settings
from .workflow import build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-tool travel planner.")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--nights", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-3.2-multi-tool-travel-planner")
    args = parse_args()
    app = build_app(load_settings())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke(
        {
            "origin": args.origin,
            "destination": args.destination,
            "nights": args.nights,
            "budget": float(args.budget),
            "budget_retry_count": 0,
        },
        config=config,
    )
    print(result["itinerary"])
    print("\nBudget note:")
    print(result["budget_note"])


if __name__ == "__main__":
    main()
