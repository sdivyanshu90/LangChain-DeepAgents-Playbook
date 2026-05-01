from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ..config import Settings


class WriterState(TypedDict, total=False):
    goal: str
    last_instruction: str
    completed_work: Annotated[list[str], operator.add]
    draft: str
    review_feedback: dict


def build_writer_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a writer. Produce or revise the current deliverable draft."),
            (
                "human",
                "Goal: {goal}\nInstruction: {last_instruction}\nCompleted work:\n{completed_work}\nCurrent draft:\n{draft}\nReview feedback:\n{review_feedback}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    chain = prompt | model | StrOutputParser()

    def write_node(state: WriterState) -> WriterState:
        result = chain.invoke(
            {
                "goal": state["goal"],
                "last_instruction": state.get("last_instruction", "Draft the deliverable."),
                "completed_work": "\n".join(state.get("completed_work", [])),
                "draft": state.get("draft", "(none yet)"),
                "review_feedback": state.get("review_feedback", {}),
            }
        )
        return {"draft": result, "completed_work": ["Writer: updated the deliverable draft."]}

    graph = StateGraph(WriterState)
    graph.add_node("write", write_node)
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    return graph.compile()