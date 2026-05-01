"""LangSmith tracing setup - call configure_tracing() at application entry point."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def configure_tracing(project_name: str) -> bool:
    """
    Enable LangSmith tracing if LANGSMITH_API_KEY is present in the environment.
    Returns True if tracing was enabled, False if the key was missing (soft failure).
    """
    load_dotenv()
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("[LangSmith] LANGSMITH_API_KEY not set - tracing disabled.")
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name
    print(f"[LangSmith] Tracing enabled -> project: {project_name}")
    return True


def setup_langsmith(project_name: str, *, enabled: bool = True) -> bool:
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    return configure_tracing(project_name)
