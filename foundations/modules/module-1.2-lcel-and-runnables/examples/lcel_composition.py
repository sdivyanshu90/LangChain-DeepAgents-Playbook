from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    parser = StrOutputParser()

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You summarize notes for busy software teams."),
            ("human", "Summarize this note in 2 bullet points:\n\n{note}"),
        ]
    )
    risks_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You identify operational risks without inventing facts."),
            ("human", "List the main risks in this note. If none exist, say 'No clear risk found.'\n\n{note}"),
        ]
    )

    analysis = RunnableParallel(
        source_note=RunnablePassthrough(),
        summary=summary_prompt | model | parser,
        risks=risks_prompt | model | parser,
    )

    result = analysis.invoke(
        "Migration is blocked because the billing team has not approved the new quota model. "
        "If approval slips again, launch will move by one week."
    )

    print(result)


if __name__ == "__main__":
    main()
