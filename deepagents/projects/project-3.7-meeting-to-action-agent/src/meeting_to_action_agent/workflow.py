from __future__ import annotations

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .config import Settings


class ActionItem(BaseModel):
    owner: str = Field(..., description="Person or team responsible for the task.")
    task: str = Field(..., description="Action item description.")
    deadline: str | None = Field(default=None, description="Deadline if explicitly present.")


class MeetingActionPackage(BaseModel):
    summary: str = Field(..., description="Short meeting summary.")
    decisions: list[str] = Field(default_factory=list, description="Meeting decisions.")
    action_items: list[ActionItem] = Field(default_factory=list, description="Action items extracted from the meeting.")
    open_questions: list[str] = Field(default_factory=list, description="Open questions remaining after the meeting.")


class MeetingState(TypedDict, total=False):
    text: str
    normalized_text: str
    action_package: dict
    draft_email: str
    human_approved: bool
    rendered_output: str


def render_action_package(action_package: dict) -> str:
    lines = ["# Meeting Action Summary", "", action_package["summary"], ""]
    if action_package["decisions"]:
        lines.extend(["## Decisions", ""])
        for decision in action_package["decisions"]:
            lines.append(f"- {decision}")
        lines.append("")
    if action_package["action_items"]:
        lines.extend(["## Action Items", ""])
        for item in action_package["action_items"]:
            deadline = item.get("deadline") or "No deadline"
            lines.append(f"- {item['owner']}: {item['task']} ({deadline})")
        lines.append("")
    if action_package["open_questions"]:
        lines.extend(["## Open Questions", ""])
        for question in action_package["open_questions"]:
            lines.append(f"- {question}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_final_output(action_package: dict, draft_email: str) -> str:
    base = render_action_package(action_package).rstrip()
    return f"{base}\n\n## Follow-Up Email\n\n{draft_email}\n"


def apply_human_review(state: MeetingState, review_response: dict) -> Command:
    if review_response.get("approved"):
        return Command(update={"human_approved": True}, goto="finalize")
    return Command(
        update={
            "draft_email": review_response.get("edited_email", state["draft_email"]),
            "human_approved": True,
        },
        goto="finalize",
    )


def build_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You extract meeting decisions, action items, and open questions from the provided text."),
            ("human", "Meeting text:\n{normalized_text}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    extractor = prompt | model.with_structured_output(MeetingActionPackage)
    email_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You draft concise follow-up emails based on meeting decisions and action items."),
            (
                "human",
                "Meeting summary package:\n{action_package}",
            ),
        ]
    )
    email_chain = email_prompt | model | StrOutputParser()

    def normalize_node(state: MeetingState) -> MeetingState:
        return {"normalized_text": " ".join(state["text"].split())}

    def extract_node(state: MeetingState) -> MeetingState:
        package = extractor.invoke({"normalized_text": state["normalized_text"]})
        return {"action_package": package.model_dump()}

    def email_draft_node(state: MeetingState) -> MeetingState:
        return {"draft_email": email_chain.invoke({"action_package": state["action_package"]})}

    def human_review_node(state: MeetingState) -> Command:
        edited = interrupt(
            {
                "draft_email": state["draft_email"],
                "action_items": state["action_package"].get("action_items", []),
                "instruction": "Review the draft. Return approved=True or provide edited_email.",
            }
        )
        return apply_human_review(state, edited)

    def finalize_node(state: MeetingState) -> MeetingState:
        return {"rendered_output": render_final_output(state["action_package"], state["draft_email"])}

    graph = StateGraph(MeetingState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("extract", extract_node)
    graph.add_node("email_draft", email_draft_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "extract")
    graph.add_edge("extract", "email_draft")
    graph.add_edge("email_draft", "human_review")
    graph.add_edge("finalize", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
