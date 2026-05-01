from pathlib import Path

from codebase_explorer.workflow import inventory_repo, route_after_scan, summarize_inventory_modules


def test_inventory_repo_lists_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')", encoding="utf-8")

    inventory = inventory_repo(str(tmp_path))
    assert "README.md" in inventory
    assert "src/app.py" in inventory


def test_summarize_inventory_modules_groups_files_by_top_level() -> None:
    summaries = summarize_inventory_modules(["README.md", "src/app.py", "src/utils.py", "tests/test_app.py"])

    assert any(summary.startswith("Module: src") for summary in summaries)
    assert any("tests/test_app.py" in summary for summary in summaries)


def test_route_after_scan_prefers_select_when_question_present() -> None:
    assert route_after_scan({"current_question": "Where are the entry points?"}) == "select"
