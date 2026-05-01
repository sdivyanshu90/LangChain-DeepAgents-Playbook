from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings


SOURCE_LIBRARY = {
    "ai support": [
        "Escalation paths reduce the damage from low-confidence automation.",
        "Evaluation loops matter because support quality is measured by resolution accuracy, not only speed.",
    ],
    "incident workflows": [
        "Operational runbooks work better when they include explicit retry and rollback criteria.",
        "Incident response systems need visible ownership, checkpoints, and recovery paths.",
    ],
}


def gather_sources(topic: str) -> list[str]:
    topic_lower = topic.lower()
    for key, items in SOURCE_LIBRARY.items():
        if key in topic_lower:
            return items
    return ["No exact source match found in the starter library."]


class ContentState(TypedDict, total=False):
    topic: str
    audience: str
    source_notes: Annotated[list[str], operator.add]
    draft: str
    review_notes: str
    review_score: int
    revision_count: int
    revised_draft: str


class ReviewResult(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality score")
    feedback: str = Field(description="Specific, actionable critique")
    approved: bool = Field(description="True if score >= 8")


def route_after_review(state: ContentState) -> str:
    score = state.get("review_score", 0)
    revision_count = state.get("revision_count", 0)
    if score >= 8 or revision_count >= 3:
        return "finalize"
    return "revise"


def build_app(settings: Settings):
    draft_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You draft concise operational content from the provided source notes."),
            ("human", "Topic: {topic}\nAudience: {audience}\nSources:\n{source_notes}"),
        ]
    )
    review_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You review drafts and return a numeric quality score plus concrete revision notes."),
            ("human", "Topic: {topic}\nAudience: {audience}\nDraft:\n{draft}"),
        ]
    )
    revise_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You revise drafts using the provided review notes while staying grounded in the original sources."),
            ("human", "Topic: {topic}\nAudience: {audience}\nSources:\n{source_notes}\nDraft:\n{draft}\nReview notes:\n{review_notes}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    drafter = draft_prompt | model | StrOutputParser()
    reviewer = review_prompt | model.with_structured_output(ReviewResult)
    reviser = revise_prompt | model | StrOutputParser()

    def source_node(state: ContentState) -> ContentState:
        return {"source_notes": gather_sources(state["topic"])}

    def draft_node(state: ContentState) -> ContentState:
        draft = drafter.invoke(
            {
                "topic": state["topic"],
                "audience": state["audience"],
                "source_notes": "\n".join(state.get("source_notes", [])),
            }
        )
        return {"draft": draft, "revision_count": 0}

    def review_node(state: ContentState) -> ContentState:
        review_result = reviewer.invoke(
            {
                "topic": state["topic"],
                "audience": state["audience"],
                "draft": state["draft"],
            }
        )
        return {"review_notes": review_result.feedback, "review_score": review_result.score}

    def revise_node(state: ContentState) -> ContentState:
        revised = reviser.invoke(
            {
                "topic": state["topic"],
                "audience": state["audience"],
                "source_notes": "\n".join(state.get("source_notes", [])),
                "draft": state["draft"],
                "review_notes": state["review_notes"],
            }
        )
        return {"draft": revised, "revision_count": state.get("revision_count", 0) + 1}

    def finalize_node(state: ContentState) -> ContentState:
        return {"revised_draft": state["draft"]}

    graph = StateGraph(ContentState)
    graph.add_node("sources", source_node)
    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "sources")
    graph.add_edge("sources", "draft")
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", route_after_review, {"finalize": "finalize", "revise": "revise"})
    graph.add_edge("revise", "review")
    graph.add_edge("finalize", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
