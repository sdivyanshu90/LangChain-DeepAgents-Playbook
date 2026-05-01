from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .bot import answer_policy_question
from .config import load_settings
from .indexer import build_retriever
from .loaders import load_policy_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer internal support questions from policy docs.")
    parser.add_argument("--policies-dir", required=True, help="Directory containing policy documents.")
    parser.add_argument("--question", required=True, help="Support question to answer.")
    parser.add_argument("--department", help="Optional department scope such as finance or security.")
    parser.add_argument("--format", choices=("json", "text"), default="text")
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from shared.utils.langsmith_setup import configure_tracing

    configure_tracing("project-2.3-support-bot")
    args = parse_args()
    settings = load_settings()
    documents = load_policy_documents(args.policies_dir)
    retriever = build_retriever(documents, settings)
    decision = answer_policy_question(
        question=args.question,
        retriever=retriever,
        settings=settings,
        department=args.department,
    )

    if args.format == "json":
        print(decision.model_dump_json(indent=2))
        return

    print(decision.answer)
    print(f"\nConfidence: {decision.confidence}")
    print(f"Escalation required: {decision.escalation_required}")

    if decision.cited_policies:
        print("\nCited policies:")
        for policy in decision.cited_policies:
            print(f"- {policy}")

    if decision.missing_information:
        print("\nMissing information:")
        for item in decision.missing_information:
            print(f"- {item}")

    if decision.escalation_reason:
        print(f"\nEscalation reason: {decision.escalation_reason}")


if __name__ == "__main__":
    main()
