from meeting_to_action_agent.workflow import apply_human_review, render_action_package, render_final_output


def test_render_action_package_outputs_sections() -> None:
    rendered = render_action_package(
        {
            "summary": "Launch planning moved by one week.",
            "decisions": ["Delay launch by one week."],
            "action_items": [{"owner": "Priya", "task": "Update rollout plan", "deadline": "Friday"}],
            "open_questions": ["Will enablement be ready in time?"],
        }
    )
    assert "## Decisions" in rendered
    assert "Priya: Update rollout plan" in rendered


def test_render_final_output_includes_follow_up_email() -> None:
    rendered = render_final_output(
        {
            "summary": "Launch planning moved by one week.",
            "decisions": ["Delay launch by one week."],
            "action_items": [],
            "open_questions": [],
        },
        "Team, here is the updated rollout plan.",
    )

    assert "## Follow-Up Email" in rendered
    assert "updated rollout plan" in rendered


def test_apply_human_review_uses_edited_email_when_provided() -> None:
    command = apply_human_review(
        {"draft_email": "Original email draft"},
        {"edited_email": "Updated human-reviewed email"},
    )

    assert command.goto == "finalize"
    assert command.update["draft_email"] == "Updated human-reviewed email"
