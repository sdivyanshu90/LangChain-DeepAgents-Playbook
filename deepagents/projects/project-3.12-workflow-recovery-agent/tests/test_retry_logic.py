from workflow_recovery_agent.config import Settings
from workflow_recovery_agent.workflow import build_app, execute_once, fallback


def test_execute_once_requests_retry_on_first_attempt() -> None:
    result = execute_once({"task": "refresh status", "retries": 0})

    assert result["status"] == "retry"
    assert result["checkpoints"] == ["attempt 1: primary step failed"]


def test_fallback_marks_degraded_mode() -> None:
    class FakeChain:
        def invoke(self, payload: dict) -> str:
            assert payload["task"] == "refresh status"
            return "Fallback summary from cached data"

    result = fallback(
        {
            "task": "refresh status",
            "partial_results": ["attempt 1 failed"],
        },
        fallback_chain=FakeChain(),
        max_retries=2,
    )

    assert result["fallback_used"] is True
    assert result["result"] == "Fallback summary from cached data"


def test_workflow_recovery_returns_result(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    app = build_app(Settings(max_retries=2))
    config = {"configurable": {"thread_id": "test-workflow-recovery"}}
    result = app.invoke(
        {
            "task": "refresh status",
            "retries": 0,
            "max_retries": 2,
            "checkpoints": [],
            "partial_results": [],
            "fallback_used": False,
        },
        config=config,
    )
    assert "result" in result
    assert len(result.get("checkpoints", [])) >= 1
    assert result.get("fallback_used") is False
