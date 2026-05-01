from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


def select_context(payload: dict) -> dict:
    state = payload["state"]
    profile = state["user_profile"]
    session = state["session"]

    return {
        "question": payload["question"],
        "user_context": (
            f"name={profile['name']}; team={profile['team']}; preferred_style={profile['preferred_style']}"
        ),
        "session_context": (
            f"current_goal={session['current_goal']}; recent_summary={session['recent_summary']}"
        ),
    }


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer using the provided user and session context. Stay focused on the current goal.",
            ),
            (
                "human",
                "User context: {user_context}\n"
                "Session context: {session_context}\n"
                "Question: {question}",
            ),
        ]
    )

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    chain = RunnableLambda(select_context) | prompt | model | StrOutputParser()

    state = {
        "user_profile": {
            "name": "Dana",
            "team": "Finance",
            "preferred_style": "bullet points",
        },
        "session": {
            "current_goal": "Understand the reimbursement policy rollout",
            "recent_summary": "The user already asked about approval steps and deadline handling.",
        },
    }

    result = chain.invoke(
        {
            "question": "What should I focus on next?",
            "state": state,
        }
    )

    print(result)


if __name__ == "__main__":
    main()
