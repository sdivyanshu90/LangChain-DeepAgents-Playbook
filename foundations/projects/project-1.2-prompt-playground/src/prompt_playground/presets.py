from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPreset:
    name: str
    system_instruction: str


PRESETS: dict[str, PromptPreset] = {
    "teacher": PromptPreset(
        name="teacher",
        system_instruction=(
            "You are an expert technical teacher. Explain ideas with clarity, precise examples, "
            "and minimal jargon."
        ),
    ),
    "reviewer": PromptPreset(
        name="reviewer",
        system_instruction=(
            "You are a rigorous reviewer. Identify weak spots, assumptions, and concrete improvements."
        ),
    ),
    "strategist": PromptPreset(
        name="strategist",
        system_instruction=(
            "You are a pragmatic strategist. Turn ambiguous requests into prioritized next steps."
        ),
    ),
}


def get_preset(name: str) -> PromptPreset:
    try:
        return PRESETS[name]
    except KeyError as error:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}.") from error
