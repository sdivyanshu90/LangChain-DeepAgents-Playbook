from customer_support_triage_agent.workflow import classify_issue, route_case


def test_classify_issue_routes_billing_cases() -> None:
    queue, urgency = classify_issue("Invoice mismatch", "Customer reports duplicate billing")
    assert queue == "billing"
    assert urgency == "medium"


def test_classify_issue_routes_security_cases() -> None:
    queue, urgency = classify_issue("Security breach", "Customer reports a credential compromise")

    assert queue == "security"
    assert urgency == "high"


def test_route_case_escalates_low_sentiment() -> None:
    command = route_case("billing", "medium", 0.2)

    assert command.goto == "escalation_agent"
    assert command.update["routed_to"] == "escalation"
