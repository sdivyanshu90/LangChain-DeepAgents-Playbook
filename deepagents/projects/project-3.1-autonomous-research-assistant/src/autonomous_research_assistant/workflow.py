from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from .config import Settings


RESEARCH_CORPUS = {
    "support-ops": "Support automation improves response speed, but quality control and escalation paths remain necessary.",
    "governance": "AI operations programs need evaluation loops, audit trails, and clear ownership for policy decisions.",
    "costs": "Automation can reduce repetitive workload, but hidden costs appear in review workflows and exception handling.",
}


def search_sources(question: str) -> list[str]:
    question_lower = question.lower()
    matches = []
    for source_id, content in RESEARCH_CORPUS.items():
        if any(word in content.lower() or word in source_id for word in question_lower.split()):
            matches.append(source_id)
    return matches or list(RESEARCH_CORPUS.keys())[:2]


class SearchPlan(BaseModel):
    sub_queries: list[str] = Field(
        description="3-5 targeted search queries that together cover the research question"
    )


class ResearchState(TypedDict, total=False):
    question: str
    search_plan: tuple[str, ...]
    raw_results: Annotated[list[str], operator.add]
    action_log: Annotated[list[str], operator.add]
    synthesis: str
    quality_score: int
    quality_feedback: str
    iteration_count: int
    final_brief: str


def fan_out_searches(state: ResearchState) -> list[Send]:
    return [
        Send("search_single", {"question": state["question"], "sub_query": sub_query})
        for sub_query in state.get("search_plan", ())
    ]


def route_after_quality(state: ResearchState) -> str:
    if state["quality_score"] >= 7 or state.get("iteration_count", 0) >= 3:
        return "write_report"
    return "plan"


def build_app(settings: Settings):
    plan_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a research strategist. Given a question, produce a search plan."),
            ("human", "Question: {question}"),
        ]
    )
    synth_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You write concise research briefs grounded in the provided evidence."),
            ("human", "Question: {question}\n\nEvidence:\n{evidence}\n\nAction log:\n{action_log}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    planner_chain = plan_prompt | model.with_structured_output(SearchPlan)
    summarizer = synth_prompt | model | StrOutputParser()
    score_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rate the completeness of this research synthesis from 1-10. "
                    "Return only a JSON object: {\"score\": <int>, \"feedback\": <str>}",
                ),
                ("human", "{synthesis}"),
            ]
        )
        | model
        | JsonOutputParser()
    )

    def plan_node(state: ResearchState) -> ResearchState:
        plan = planner_chain.invoke({"question": state["question"]})
        return {
            "search_plan": tuple(plan.sub_queries),
            "action_log": [f"planned {len(plan.sub_queries)} search step(s)"],
        }

    def search_single_node(state: dict) -> ResearchState:
        sources = search_sources(state["sub_query"])
        results = [f"[{source}] {RESEARCH_CORPUS[source]}" for source in sources]
        return {
            "raw_results": results,
            "action_log": [f"searched '{state['sub_query']}' and found {len(results)} result(s)"],
        }

    def synthesize_node(state: ResearchState) -> ResearchState:
        synthesis = summarizer.invoke(
            {
                "question": state["question"],
                "evidence": "\n\n".join(state.get("raw_results", [])),
                "action_log": "\n".join(state.get("action_log", [])),
            }
        )
        return {"synthesis": synthesis}

    def quality_gate_node(state: ResearchState) -> ResearchState:
        result = score_chain.invoke({"synthesis": state["synthesis"]})
        score = int(result["score"])
        feedback = str(result.get("feedback", ""))
        return {
            "quality_score": score,
            "quality_feedback": feedback,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "action_log": [f"quality gate scored synthesis at {score}/10"],
        }

    def write_report_node(state: ResearchState) -> ResearchState:
        return {"final_brief": state["synthesis"]}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("search_single", search_single_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("write_report", write_report_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out_searches, ["search_single"])
    graph.add_edge("search_single", "synthesize")
    graph.add_edge("synthesize", "quality_gate")
    graph.add_conditional_edges("quality_gate", route_after_quality, {"write_report": "write_report", "plan": "plan"})
    graph.add_edge("write_report", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
