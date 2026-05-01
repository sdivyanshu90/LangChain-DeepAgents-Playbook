from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings


def extract_search_contents(results: list[dict]) -> list[str]:
    extracted = []
    for item in results:
        content = item.get("content") if isinstance(item, dict) else None
        if content:
            extracted.append(content)
    return extracted or ["No results returned from Tavily."]


class SalesBrief(BaseModel):
    company_summary: str = Field(..., description="Short description of the account.")
    recent_signals: list[str] = Field(default_factory=list, description="Recent developments relevant to outreach.")
    likely_priorities: list[str] = Field(default_factory=list, description="Likely priorities inferred from the company context.")
    conversation_hooks: list[str] = Field(default_factory=list, description="Practical outreach hooks for the sales team.")


class SalesState(TypedDict, total=False):
    company: str
    search_results: Annotated[list[str], operator.add]
    news_items: Annotated[list[str], operator.add]
    draft_brief: dict
    quality_scores: dict[str, int]
    recheck_count: int
    final_brief: dict


def route_after_quality(state: SalesState) -> str:
    low_dims = [dimension for dimension, score in state["quality_scores"].items() if score < 6]
    if not low_dims or state.get("recheck_count", 0) >= 2:
        return "write_brief"
    return "recheck"


class QualityAssessment(BaseModel):
    company_summary: int = Field(ge=1, le=10)
    recent_signals: int = Field(ge=1, le=10)
    likely_priorities: int = Field(ge=1, le=10)
    conversation_hooks: int = Field(ge=1, le=10)


def build_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You create concise, grounded sales intelligence briefs from the provided company context and signals."),
            ("human", "Company: {company}\nProfile research:\n{search_results}\nRecent news:\n{news_items}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    search = TavilySearchResults(max_results=5)
    drafter = prompt | model.with_structured_output(SalesBrief)
    quality_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Score the sales brief from 1 to 10 for each dimension: company_summary, recent_signals, likely_priorities, conversation_hooks.",
            ),
            ("human", "Brief:\n{brief_json}"),
        ]
    )
    quality_chain = quality_prompt | model.with_structured_output(QualityAssessment)

    def profile_node(state: SalesState) -> SalesState:
        results = search.invoke(f"{state['company']} company overview products revenue")
        return {"search_results": extract_search_contents(results)}

    def news_node(state: SalesState) -> SalesState:
        results = search.invoke(f"{state['company']} latest news 2024 2025")
        return {"news_items": extract_search_contents(results)}

    def synthesize_node(state: SalesState) -> SalesState:
        brief = drafter.invoke(
            {
                "company": state["company"],
                "search_results": "\n".join(state.get("search_results", [])),
                "news_items": "\n".join(state.get("news_items", [])),
            }
        )
        return {"draft_brief": brief.model_dump()}

    def quality_check_node(state: SalesState) -> SalesState:
        assessment = quality_chain.invoke({"brief_json": state["draft_brief"]})
        return {
            "quality_scores": assessment.model_dump(),
            "recheck_count": state.get("recheck_count", 0) + 1,
        }

    def write_brief_node(state: SalesState) -> SalesState:
        return {"final_brief": state["draft_brief"]}

    graph = StateGraph(SalesState)
    graph.add_node("profile", profile_node)
    graph.add_node("news", news_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("write_brief", write_brief_node)
    graph.add_edge(START, "profile")
    graph.add_edge("profile", "news")
    graph.add_edge("news", "synthesize")
    graph.add_edge("synthesize", "quality_check")
    graph.add_conditional_edges("quality_check", route_after_quality, {"write_brief": "write_brief", "recheck": "synthesize"})
    graph.add_edge("write_brief", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
