import pytest

from prompt_playground.chain import build_chain
from prompt_playground.config import Settings
from prompt_playground.presets import PRESETS, get_preset


def test_known_preset_is_returned() -> None:
    preset = get_preset("teacher")

    assert preset.name == "teacher"
    assert "technical teacher" in preset.system_instruction


def test_all_expected_presets_exist() -> None:
    assert {"teacher", "reviewer", "strategist"}.issubset(PRESETS)


def test_invalid_output_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported output mode"):
        build_chain(
            settings=Settings(model="gpt-4.1-mini", temperature=0.0),
            preset_name="teacher",
            output_mode="invalid",
        )
