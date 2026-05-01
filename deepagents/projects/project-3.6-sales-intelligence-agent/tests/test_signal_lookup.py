from sales_intelligence_agent.workflow import extract_search_contents, route_after_quality


def test_extract_search_contents_returns_content_values() -> None:
    results = extract_search_contents([
        {"content": "Northwind opened a new APAC partnership."},
        {"content": "Northwind highlighted deployment uptime improvements."},
    ])

    assert len(results) == 2


def test_route_after_quality_requests_recheck_for_low_scores() -> None:
    route = route_after_quality(
        {
            "quality_scores": {
                "company_summary": 8,
                "recent_signals": 5,
                "likely_priorities": 8,
                "conversation_hooks": 7,
            },
            "recheck_count": 1,
        }
    )

    assert route == "recheck"


def test_route_after_quality_finishes_when_scores_are_good() -> None:
    route = route_after_quality(
        {
            "quality_scores": {
                "company_summary": 8,
                "recent_signals": 7,
                "likely_priorities": 8,
                "conversation_hooks": 7,
            },
            "recheck_count": 1,
        }
    )

    assert route == "write_brief"
