from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_chain(model_name: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You explain technical notes with concrete, beginner-friendly language."),
            ("human", "Explain this note in plain English and keep it under 120 words:\n\n{notes}"),
        ]
    )

    model = ChatOpenAI(model=model_name, temperature=0)
    return prompt | model | StrOutputParser()


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    chain = build_chain(model_name)

    notes = "LCEL lets you compose prompts, models, and parsers into a readable pipeline."
    result = chain.invoke({"notes": notes})
    print(result)


if __name__ == "__main__":
    main()
