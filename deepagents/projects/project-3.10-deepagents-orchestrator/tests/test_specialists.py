from deepagents_orchestrator.supervisor import AGENTS
from deepagents_orchestrator.workflow import route_after_review


def test_supervisor_agent_list_includes_finish() -> None:
    assert "FINISH" in AGENTS


def test_route_after_review_loops_to_writer_for_low_scores() -> None:
    route = route_after_review({"review_feedback": {"score": 6}, "revision_count": 1})

    assert route == "writer"


def test_route_after_review_returns_supervisor_for_high_scores() -> None:
    route = route_after_review({"review_feedback": {"score": 9}, "revision_count": 1})

    assert route == "supervisor"


def test_route_after_review_honors_revision_cap() -> None:
    route = route_after_review({"review_feedback": {"score": 4}, "revision_count": 3})

    assert route == "supervisor"
