from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    max_retries: int = 2
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0


def load_settings() -> Settings:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")
    return Settings(
        max_retries=int(os.getenv("WORKFLOW_RECOVERY_MAX_RETRIES", "2")),
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
    )
