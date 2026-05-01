"""
shared/utils/model_factory.py
────────────────────────────────────────────────────────────────
Central factory for instantiating LLM clients used throughout the curriculum.

WHY THIS EXISTS
───────────────
Every project needs a language model client. If each project hard-codes its own
ChatOpenAI(...) call with different parameter names, three problems emerge:

  1. Model upgrades require editing every file.
  2. Developers forget to wire temperature, streaming, or timeout consistently.
  3. Switching providers (OpenAI ↔ Anthropic ↔ local) requires structural changes.

A central factory solves all three: one change propagates everywhere, and
provider selection becomes a config flag, not a code change.

SUPPORTED PROVIDERS
───────────────────
  "openai"     → ChatOpenAI     (default)
  "anthropic"  → ChatAnthropic
  "google"     → ChatGoogleGenerativeAI
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

Provider = Literal["openai", "anthropic", "google"]


def get_chat_model(
    provider: Provider | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    streaming: bool = False,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Return a configured chat model for the requested provider.

    Parameters
    ----------
    provider : str | None
        One of "openai", "anthropic", "google".
        Defaults to the MODEL_PROVIDER env var, then falls back to "openai".
    model : str | None
        Specific model name. If None, uses the provider-specific env var
        (OPENAI_MODEL, ANTHROPIC_MODEL, GOOGLE_MODEL) or a sensible default.
    temperature : float
        Sampling temperature. Use 0.0 for deterministic outputs in agents.
    streaming : bool
        Enable token streaming. Required for real-time UIs (Streamlit, etc.).
    max_tokens : int | None
        Hard cap on output tokens. None means use the provider default.

    Returns
    -------
    BaseChatModel
        A configured LangChain chat model instance.

    Raises
    ------
    ValueError
        If the provider is not recognised or the required API key is missing.
    """
    resolved_provider: str = (
        provider
        or os.getenv("MODEL_PROVIDER", "openai")
    ).lower()

    if resolved_provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        return ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            streaming=streaming,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    if resolved_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        return ChatAnthropic(
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            temperature=temperature,
            streaming=streaming,
            max_tokens=max_tokens or 4096,
            api_key=api_key,
        )

    if resolved_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
            streaming=streaming,
            google_api_key=api_key,
        )

    raise ValueError(
        f"Unknown provider '{resolved_provider}'. "
        "Valid options: 'openai', 'anthropic', 'google'."
    )
