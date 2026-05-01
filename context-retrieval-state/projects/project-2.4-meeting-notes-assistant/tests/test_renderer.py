from langchain_core.documents import Document

from meeting_notes_assistant.config import Settings
from meeting_notes_assistant.renderer import render_markdown
from meeting_notes_assistant.schemas import ActionItem, MeetingSummary
from meeting_notes_assistant.summarizer import summarize_document, summarize_documents


def test_render_markdown_contains_decisions_and_actions() -> None:
    summaries = [
        MeetingSummary(
            source="notes/launch_sync.md",
            meeting_title="Launch Sync",
            summary="The team aligned on launch readiness blockers.",
            decisions=["Delay launch by one week until finance approval."],
            action_items=[ActionItem(owner="Alex", task="Confirm finance approval date")],
            open_questions=["Will enablement be ready by the new date?"],
        )
    ]

    rendered = render_markdown(summaries)

    assert "## Launch Sync" in rendered
    assert "### Decisions" in rendered
    assert "Alex: Confirm finance approval date" in rendered


def test_summarize_document_invokes_chain_with_metadata() -> None:
    document = Document(
        page_content="We delayed launch and asked Alex to confirm finance approval.",
        metadata={"source": "notes/launch_sync.md", "title": "Launch Sync"},
    )

    class FakeChain:
        def invoke(self, payload: dict) -> MeetingSummary:
            assert payload["source"] == "notes/launch_sync.md"
            assert payload["title"] == "Launch Sync"
            return MeetingSummary(
                source=payload["source"],
                meeting_title=payload["title"],
                summary="Launch timing changed.",
                decisions=["Delay launch by one week."],
                action_items=[ActionItem(owner="Alex", task="Confirm finance approval")],
                open_questions=["Is enablement ready?"],
            )

    summary = summarize_document(document, FakeChain())

    assert summary.decisions == ["Delay launch by one week."]
    assert summary.action_items[0].owner == "Alex"


def test_summarize_documents_returns_empty_list_for_empty_input() -> None:
    summaries = summarize_documents(
        [],
        Settings(model="gpt-4.1-mini", embedding_model="text-embedding-3-small", temperature=0.0),
        chain=object(),
    )

    assert summaries == []
