from pathlib import Path

from docuchat.cli import reset_persist_dir
from docuchat.session import FileChatSessionStore


def test_file_session_store_appends_and_loads_messages(tmp_path: Path) -> None:
    store = FileChatSessionStore(str(tmp_path))
    store.append_turn("demo", "hello", "hi there")
    store.append_turn("demo", "second", "reply")

    messages = store.load_messages("demo")

    assert len(messages) == 4
    assert str(messages[0].content) == "hello"
    assert str(messages[-1].content) == "reply"


def test_file_session_store_respects_limit(tmp_path: Path) -> None:
    store = FileChatSessionStore(str(tmp_path))
    store.append_turn("demo", "hello", "hi there")
    store.append_turn("demo", "second", "reply")

    messages = store.load_messages("demo", limit=2)

    assert len(messages) == 2
    assert str(messages[0].content) == "second"


def test_reset_persist_dir_removes_existing_directory(tmp_path: Path) -> None:
    persist_dir = tmp_path / ".chroma_docuchat"
    persist_dir.mkdir()
    (persist_dir / "marker.txt").write_text("indexed", encoding="utf-8")

    reset_persist_dir(str(persist_dir))

    assert not persist_dir.exists()
