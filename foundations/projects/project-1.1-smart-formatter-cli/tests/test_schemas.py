from smart_formatter.chain import format_note
from smart_formatter.config import Settings
from smart_formatter.schemas import ActionItem, StructuredNote


def test_structured_note_defaults() -> None:
    record = StructuredNote(topic="Budget review", summary="Finance asked for an updated draft.")

    assert record.key_points == []
    assert record.action_items == []
    assert record.risks == []
    assert record.follow_up_questions == []


def test_action_item_keeps_deadline_optional() -> None:
    item = ActionItem(owner="Alex", task="Send revised numbers")

    assert item.deadline is None


def test_format_note_returns_mocked_structured_output(monkeypatch) -> None:
    class FakeFormatter:
        def invoke(self, payload: dict) -> StructuredNote:
            assert payload["raw_text"] == "Messy note"
            return StructuredNote(
                topic="Launch review",
                summary="The team reviewed launch blockers.",
                key_points=["Two issues remain open."],
                action_items=[ActionItem(owner="Dana", task="Confirm staging fix")],
                risks=["Timeline slip"],
                follow_up_questions=["Who owns rollback approval?"],
            )

    monkeypatch.setattr("smart_formatter.chain.build_formatter", lambda settings: FakeFormatter())

    result = format_note("  Messy note  ", Settings(model="gpt-4.1-mini", temperature=0.0))

    assert result.topic == "Launch review"
    assert result.action_items[0].owner == "Dana"
    assert result.risks == ["Timeline slip"]
