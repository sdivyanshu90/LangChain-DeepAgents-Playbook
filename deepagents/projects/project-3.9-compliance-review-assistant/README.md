# Project 3.9: Compliance Review Assistant

Detect policy risks, missing information, and escalation needs in a structured review workflow.

## Learning Objective

Learn how to build a compliance-oriented workflow that compares a submission against policy criteria, records risks, and escalates when evidence is incomplete.

## Real-World Use Case

Teams often need a first-pass review of launch plans, vendor requests, or process changes against internal policy. A useful assistant should flag risks and missing information instead of giving a false pass.

## Difficulty

Advanced

## Skills Covered

- policy-aware workflow design
- structured risk reporting
- explicit missing-information handling
- escalation-aware output design
- graph-based review control

## Architecture Overview

The workflow has three stages:

1. Load policy checklist context.
2. Review the submission against that policy.
3. Return a structured compliance report.

Why this design matters:

- policy criteria remain visible inputs
- missing information is a first-class output
- escalation can be triggered explicitly rather than implied

## Input and Output Expectations

Input:

- policy topic
- submission text

Output:

- compliance summary
- identified risks
- missing information list
- escalation recommendation

## Dependencies and Setup

```bash
cd deepagents/projects/project-3.9-compliance-review-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## File Structure

```text
project-3.9-compliance-review-assistant/
├── .env.example
├── README.md
├── requirements.txt
├── src/
│   └── compliance_review_assistant/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── workflow.py
└── tests/
    └── test_policy_lookup.py
```

## Step-by-Step Build Instructions

### 1. Model the policy checklist explicitly

The policy criteria should not live only inside a prompt.

### 2. Keep review output structured

Risk review is easier to act on when the result has stable fields.

### 3. Treat missing information as a meaningful outcome

Many compliance decisions fail because information is absent, not because it is obviously wrong.

### 4. Add escalation explicitly

The assistant should say when human review is required.

## Core Implementation Code

```bash
PYTHONPATH=src python -m compliance_review_assistant.cli \
  --policy-topic "vendor_access" \
  --submission "The vendor will access production data during onboarding and provide a security questionnaire later."
```

## Important Design Choices

### Why keep checklist data outside the prompt?

Because policy criteria should be inspectable and reusable.

### Why include missing information explicitly?

Because compliance review often depends on absent details.

### Why make escalation a field rather than a tone choice?

Because operational systems need a stable signal they can route on.

## Common Pitfalls

- turning a policy review into a vague summary
- marking a case safe when critical data is missing
- hiding escalation need behind soft language
- treating the prompt as the only policy source

## Testing or Validation Approach

Use syntax validation first:

```bash
python -m compileall src tests
```

Then test with both complete and incomplete submissions to confirm the assistant distinguishes policy risk from missing information.

## Extension Ideas

- add policy versioning
- attach exact checklist evidence to every risk
- support multiple policy packs in one run
- add review confidence scoring

## Optional LangSmith Instrumentation

Tracing is useful here because policy loading, review reasoning, and escalation output should be visible in one workflow trace.

## Optional Deployment or Packaging Notes

This project can support vendor review, launch governance, and internal risk screening workflows.
