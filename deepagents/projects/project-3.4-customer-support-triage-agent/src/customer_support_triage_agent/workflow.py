from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from .config import Settings


def classify_issue(title: str, description: str) -> tuple[str, Literal["low", "medium", "high"]]:
    combined = f"{title} {description}".lower()
    if "security" in combined or "breach" in combined:
        return "security", "high"
    if "invoice" in combined or "billing" in combined or "charge" in combined:
        return "billing", "medium"
    return "product", "medium" if "bug" in combined or "error" in combined else "low"


class SupportTriage(BaseModel):
    queue: str = Field(..., description="Assigned support queue.")
    urgency: Literal["low", "medium", "high"] = Field(..., description="Urgency level for the case.")
    summary: str = Field(..., description="Short case summary for the assigned team.")
    next_step: str = Field(..., description="Recommended next operational action.")


class SupportState(TypedDict, total=False):
    title: str
    description: str
    queue: str
    urgency: Literal["low", "medium", "high"]
    routed_to: str
    sentiment_score: float
    summary: str
    triage_result: dict


def route_case(queue: str, urgency: Literal["low", "medium", "high"], sentiment: float) -> Command:
    if sentiment < 0.3:
        return Command(
            goto="escalation_agent",
            update={
                "queue": "escalation",
                "urgency": "high",
                "routed_to": "escalation",
                "sentiment_score": sentiment,
            },
        )
    return Command(
        goto=f"{queue}_agent",
        update={
            "queue": queue,
            "urgency": urgency,
            "routed_to": queue,
            "sentiment_score": sentiment,
        },
    )


def build_app(settings: Settings):
    sentiment_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rate customer sentiment from 0.0 (furious) to 1.0 (happy). Return only a float.",
            ),
            ("human", "{message}"),
        ]
    )
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You summarize support cases for internal teams in one concise paragraph."),
            ("human", "Queue: {queue}\nUrgency: {urgency}\nTitle: {title}\nDescription: {description}"),
        ]
    )
    next_step_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You recommend the next best support action based on the routed case."),
            ("human", "Queue: {queue}\nUrgency: {urgency}\nSummary: {summary}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    sentiment_chain = sentiment_prompt | model | StrOutputParser()
    summary_chain = summary_prompt | model | StrOutputParser()
    next_step_chain = next_step_prompt | model | StrOutputParser()

    def triage_node(state: SupportState) -> Command:
        queue, urgency = classify_issue(state["title"], state["description"])
        raw_sentiment = sentiment_chain.invoke({"message": state["description"]}).strip()
        sentiment = float(raw_sentiment)
        return route_case(queue, urgency, sentiment)

    def summarize_case(state: SupportState) -> SupportState:
        summary = summary_chain.invoke(
            {
                "queue": state["queue"],
                "urgency": state["urgency"],
                "title": state["title"],
                "description": state["description"],
            }
        )
        return {"summary": summary}

    def billing_agent(state: SupportState) -> SupportState:
        return summarize_case(state)

    def security_agent(state: SupportState) -> SupportState:
        return summarize_case(state)

    def product_agent(state: SupportState) -> SupportState:
        return summarize_case(state)

    def escalation_agent(state: SupportState) -> SupportState:
        return summarize_case(state)

    def next_step_node(state: SupportState) -> SupportState:
        next_step = next_step_chain.invoke(
            {
                "queue": state["queue"],
                "urgency": state["urgency"],
                "summary": state["summary"],
            }
        )
        triage = SupportTriage(
            queue=state["queue"],
            urgency=state["urgency"],
            summary=state["summary"],
            next_step=next_step,
        )
        return {"triage_result": triage.model_dump()}

    graph = StateGraph(SupportState)
    graph.add_node("triage", triage_node)
    graph.add_node("billing_agent", billing_agent)
    graph.add_node("security_agent", security_agent)
    graph.add_node("product_agent", product_agent)
    graph.add_node("escalation_agent", escalation_agent)
    graph.add_node("recommend", next_step_node)
    graph.add_edge(START, "triage")
    graph.add_edge("billing_agent", "recommend")
    graph.add_edge("security_agent", "recommend")
    graph.add_edge("product_agent", "recommend")
    graph.add_edge("escalation_agent", "recommend")
    graph.add_edge("recommend", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
