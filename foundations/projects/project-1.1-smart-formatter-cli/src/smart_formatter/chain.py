from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .schemas import StructuredNote


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You convert messy operational notes into grounded structured data. "
            "Only include details that are supported by the source text. "
            "If a field is missing, return an empty list or null where appropriate.",
        ),
        (
            "human",
            "Normalize this note into the requested schema.\n\n"
            "Messy note:\n{raw_text}",
        ),
    ]
)

STREAM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You convert messy operational notes into grounded JSON. "
            "Return valid JSON with keys topic, summary, key_points, action_items, risks, and follow_up_questions. "
            "Each action item must contain owner, task, and deadline.",
        ),
        (
            "human",
            "Normalize this note into JSON.\n\n"
            "Messy note:\n{raw_text}",
        ),
    ]
)


def build_formatter(settings: Settings):
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    return PROMPT | model.with_structured_output(StructuredNote)


def build_streaming_formatter(settings: Settings):
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    return STREAM_PROMPT | model | StrOutputParser()


def format_note(raw_text: str, settings: Settings) -> StructuredNote:
    cleaned_text = raw_text.strip()
    if not cleaned_text:
        raise ValueError("Input text cannot be empty.")

    formatter = build_formatter(settings)
    return formatter.invoke({"raw_text": cleaned_text})


def stream_format(raw_text: str, settings: Settings):
    """Yield streamed formatter chunks for real-time CLI output."""
    cleaned_text = raw_text.strip()
    if not cleaned_text:
        raise ValueError("Input text cannot be empty.")

    formatter = build_streaming_formatter(settings)
    for chunk in formatter.stream({"raw_text": cleaned_text}):
        yield chunk
