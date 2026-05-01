from autonomous_research_assistant.workflow import fan_out_searches, route_after_quality, search_sources


def test_search_sources_returns_matching_ids() -> None:
    results = search_sources("governance and policy decisions")
    assert "governance" in results


def test_fan_out_searches_returns_one_send_per_subquery() -> None:
    sends = fan_out_searches({"question": "research", "search_plan": ("governance", "support")})

    assert len(sends) == 2
    assert sends[0].node == "search_single"
    assert sends[0].arg["sub_query"] == "governance"


def test_route_after_quality_loops_when_score_is_low() -> None:
    assert route_after_quality({"quality_score": 5, "iteration_count": 1}) == "plan"


def test_route_after_quality_finishes_at_iteration_cap() -> None:
    assert route_after_quality({"quality_score": 5, "iteration_count": 3}) == "write_report"
