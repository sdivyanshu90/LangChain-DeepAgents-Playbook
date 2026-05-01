"""
shared/utils/logging_config.py
────────────────────────────────────────────────────────────────
Structured logging configuration for all projects in the curriculum.

WHY THIS EXISTS
───────────────
print() statements work fine for demos. In production — or in a multi-node
LangGraph workflow — they:
  - mix with library output
  - lose timestamps and level context
  - cannot be filtered or redirected to a log aggregator

This module configures Python's built-in logging with:
  - ISO-format timestamps
  - Module-qualified logger names
  - Configurable log level via LOG_LEVEL env var
  - Optional JSON mode for log-aggregator-compatible output
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects for structured log sinks."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    *,
    level: str | None = None,
    json_mode: bool = False,
) -> logging.Logger:
    """
    Configure root logger and return a named logger for the calling module.

    Parameters
    ----------
    level : str | None
        One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        Defaults to LOG_LEVEL env var, then "INFO".
    json_mode : bool
        If True, emit structured JSON lines suitable for log aggregators.

    Returns
    -------
    logging.Logger
        A logger named after the calling script.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler(sys.stdout)

    if json_mode:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    # Reconfigure the root logger — idempotent if called multiple times
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # Suppress noisy third-party loggers at WARNING level
    for noisy in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("langchain_playbook")
