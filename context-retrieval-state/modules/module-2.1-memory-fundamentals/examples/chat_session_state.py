from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return SESSION_STORE.setdefault(session_id, InMemoryChatMessageHistory())


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant that uses prior conversation context when it is relevant."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)

    conversation = RunnableWithMessageHistory(
        prompt | model,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "demo-session"}}

    first = conversation.invoke(
        {"question": "My name is Priya and I work on support operations. Please remember that."},
        config=config,
    )
    second = conversation.invoke(
        {"question": "What do you remember about me?"},
        config=config,
    )

    print(first.content)
    print("---")
    print(second.content)


if __name__ == "__main__":
    main()
