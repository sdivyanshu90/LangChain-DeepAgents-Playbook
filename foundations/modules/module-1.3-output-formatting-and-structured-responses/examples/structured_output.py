from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class StudyCard(BaseModel):
    concept: str = Field(..., description="The core topic being taught.")
    explanation: str = Field(..., description="A short beginner-friendly explanation of the concept.")
    practice_question: str = Field(..., description="One question the learner should answer next.")


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You create concise study cards for software engineers learning AI systems."),
            ("human", "Create a study card for this topic:\n\n{topic}"),
        ]
    )

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    chain = prompt | model.with_structured_output(StudyCard)
    result = chain.invoke({"topic": "Why structured outputs are safer than free-form model text."})

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
