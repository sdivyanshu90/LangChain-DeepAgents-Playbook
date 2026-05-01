from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .presets import get_preset
from .schemas import OutlineResponse


OUTPUT_MODES = ("text", "json-outline")


def build_chain(*, settings: Settings, preset_name: str, output_mode: str):
    if output_mode not in OUTPUT_MODES:
        allowed = ", ".join(OUTPUT_MODES)
        raise ValueError(f"Unsupported output mode '{output_mode}'. Allowed modes: {allowed}.")

    preset = get_preset(preset_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", preset.system_instruction),
            (
                "human",
                "Task: {task}\n"
                "Audience: {audience}\n"
                "Desired tone: {tone}\n\n"
                "Answer in a way that matches the selected preset.",
            ),
        ]
    )

    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)

    if output_mode == "json-outline":
        return prompt | model.with_structured_output(OutlineResponse)

    if output_mode == "text":
        return prompt | model | StrOutputParser()


def run_prompt(
    *,
    task: str,
    audience: str,
    tone: str,
    preset_name: str,
    output_mode: str,
    settings: Settings,
):
    if not task.strip():
        raise ValueError("Task cannot be empty.")

    chain = build_chain(settings=settings, preset_name=preset_name, output_mode=output_mode)
    return chain.invoke(
        {
            "task": task.strip(),
            "audience": audience.strip(),
            "tone": tone.strip(),
        }
    )
