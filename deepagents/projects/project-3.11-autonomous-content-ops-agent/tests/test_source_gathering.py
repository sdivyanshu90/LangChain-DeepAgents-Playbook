from autonomous_content_ops_agent.workflow import gather_sources, route_after_review


def test_gather_sources_returns_topic_matches() -> None:
    notes = gather_sources("AI support workflows")
    assert len(notes) >= 1


def test_gather_sources_returns_fallback_message_for_unknown_topics() -> None:
    notes = gather_sources("Unmapped topic")

    assert notes == ["No exact source match found in the starter library."]


def test_route_after_review_requests_revision_for_low_scores() -> None:
    assert route_after_review({"review_score": 6, "revision_count": 1}) == "revise"


def test_route_after_review_finalizes_when_revision_cap_is_hit() -> None:
    assert route_after_review({"review_score": 4, "revision_count": 3}) == "finalize"
