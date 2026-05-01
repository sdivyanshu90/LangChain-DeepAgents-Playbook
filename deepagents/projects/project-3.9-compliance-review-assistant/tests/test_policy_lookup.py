from compliance_review_assistant.workflow import calculate_risk_score, lookup_policy, required_sections_for


def test_lookup_policy_returns_known_checklist() -> None:
    policy = lookup_policy("vendor_access")
    assert "security questionnaire" in policy


def test_required_sections_for_vendor_access_returns_expected_items() -> None:
    sections = required_sections_for("vendor_access")

    assert "security questionnaire" in sections
    assert "production access approval" in sections


def test_calculate_risk_score_caps_at_100() -> None:
    score = calculate_risk_score(
        [
            {"severity": "critical", "description": "Missing security questionnaire"},
            {"severity": "critical", "description": "No production approval"},
            {"severity": "critical", "description": "Undefined data access scope"},
            {"severity": "major", "description": "Rollback plan absent"},
        ],
        ["customer communication plan", "owner"],
    )

    assert score == 100
