from multi_tool_travel_planner.workflow import estimate_budget, fetch_options, route_after_budget


def test_estimate_budget_reports_status() -> None:
    note = estimate_budget([], [], budget_cap=1200, nights=3)
    assert "budget cap" in note


def test_fetch_options_returns_flights_and_hotels() -> None:
    result = fetch_options(
        {
            "origin": "SFO",
            "destination": "NYC",
            "nights": 2,
            "budget_retry_count": 0,
        }
    )

    assert len(result["flights"]) == 2
    assert len(result["hotels"]) == 2


def test_route_after_budget_replans_when_over_budget() -> None:
    assert route_after_budget({"total_estimated_cost": 1500.0, "budget": 1000.0, "budget_retry_count": 0}) == "replan"


def test_route_after_budget_builds_itinerary_when_within_budget() -> None:
    assert route_after_budget({"total_estimated_cost": 900.0, "budget": 1000.0, "budget_retry_count": 0}) == "build_itinerary"
