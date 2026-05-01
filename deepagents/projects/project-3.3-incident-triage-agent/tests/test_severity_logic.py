from incident_triage_agent.workflow import assess_severity, escalation_gate, lookup_runbook, write_report


def test_assess_severity_detects_high_urgency() -> None:
    severity = assess_severity("Checkout outage", "Timeouts across payment path")
    assert severity == "sev1"


def test_lookup_runbook_prefers_latency_runbook() -> None:
    runbook = lookup_runbook("Latency alert", "p95 latency is climbing quickly")

    assert "Latency Degradation Runbook" in runbook


def test_escalation_gate_skips_non_sev1_incidents() -> None:
    command = escalation_gate(
        {
            "severity": "sev2",
            "root_cause_hypothesis": "queue saturation",
            "remediation_steps": ["scale workers"],
        }
    )

    assert command.goto == "write_report"
    assert command.update == {"escalation_approved": False}


def test_write_report_includes_escalation_flag() -> None:
    report = write_report(
        {
            "severity": "sev1",
            "root_cause_hypothesis": "payment path outage",
            "remediation_steps": ["rollback deployment"],
            "runbook_used": "API Error Spike Runbook",
            "escalation_approved": True,
        }
    )

    assert report["final_decision"]["escalation_approved"] is True
