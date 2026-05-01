from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..config import Settings


class ReviewFeedback(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall draft quality score.")
    feedback: str = Field(description="Specific revision advice.")


class ReviewerState(TypedDict, total=False):
    goal: str
    draft: str
    review_feedback: dict
    revision_count: int
    completed_work: Annotated[list[str], operator.add]


def build_reviewer_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a reviewer. Score the draft and return concrete revision advice."),
            ("human", "Goal: {goal}\nDraft:\n{draft}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = prompt | model.with_structured_output(ReviewFeedback)

    def review_node(state: ReviewerState) -> ReviewerState:
        result = chain.invoke({"goal": state["goal"], "draft": state.get("draft", "")})
        return {
            "review_feedback": result.model_dump(),
            "revision_count": state.get("revision_count", 0) + 1,
            "completed_work": [f"Reviewer: scored the draft {result.score}/10."],
        }

    graph = StateGraph(ReviewerState)
    graph.add_node("review", review_node)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile()