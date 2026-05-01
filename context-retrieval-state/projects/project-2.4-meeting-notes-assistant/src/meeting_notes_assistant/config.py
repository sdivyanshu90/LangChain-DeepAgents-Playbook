from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model: str
    embedding_model: str
    temperature: float


def load_settings() -> Settings:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key."
        )

    return Settings(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
    )
