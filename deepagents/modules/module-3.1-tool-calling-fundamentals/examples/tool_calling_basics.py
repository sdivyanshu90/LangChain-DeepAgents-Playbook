from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def lookup_team_calendar(team_name: str) -> str:
    """Return a short summary of the team's upcoming schedule."""
    calendars = {
        "support": "Support has a backlog review on Tuesday and a training session on Thursday.",
        "finance": "Finance has a budget approval meeting on Wednesday morning.",
    }
    return calendars.get(team_name.lower(), "No schedule information was found for that team.")


@tool
def lookup_policy(topic: str) -> str:
    """Return a short policy summary for a given operational topic."""
    policies = {
        "travel": "Conference travel requires manager approval before booking.",
        "security": "Security incidents must be reported within one business day.",
    }
    return policies.get(topic.lower(), "No policy summary was found for that topic.")


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    tool_enabled_model = model.bind_tools([lookup_team_calendar, lookup_policy])
    response = tool_enabled_model.invoke(
        "Can you check whether finance has anything scheduled before I plan a budget review?"
    )

    print(response)


if __name__ == "__main__":
    main()
