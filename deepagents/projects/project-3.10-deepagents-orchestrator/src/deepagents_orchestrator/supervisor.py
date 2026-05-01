from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from .config import Settings


AGENTS = ["researcher", "analyst", "writer", "reviewer", "FINISH"]


class SupervisorDecision(BaseModel):
    next_agent: str = Field(description=f"One of: {AGENTS}")
    instructions: str = Field(description="Specific instructions for the chosen agent")


def build_supervisor_node(settings: Settings):
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    supervisor_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an orchestrator. Given the goal and work completed so far, decide which specialist to call next. "
                    "When the deliverable is complete, choose FINISH.\n\nAvailable agents: {agents}",
                ),
                (
                    "human",
                    "Goal: {goal}\n\nCompleted work:\n{completed_work}\n\nCurrent deliverable draft:\n{draft}",
                ),
            ]
        )
        | model.with_structured_output(SupervisorDecision)
    )

    def supervisor_node(state) -> Command:
        decision = supervisor_chain.invoke(
            {
                "agents": ", ".join(AGENTS),
                "goal": state["goal"],
                "completed_work": "\n".join(state.get("completed_work", [])),
                "draft": state.get("draft", "(none yet)"),
            }
        )
        if decision.next_agent == "FINISH":
            return Command(update={"final_deliverable": state.get("draft", "")}, goto="__end__")
        return Command(update={"last_instruction": decision.instructions}, goto=decision.next_agent)

    return supervisor_node