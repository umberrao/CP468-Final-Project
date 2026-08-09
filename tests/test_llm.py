import pytest

from src.llm import (
    EXPERIMENT_SETTINGS,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_ROW_NUMBERS,
    build_messages,
    parse_setting,
    prompt_record,
    setting_name,
    user_prompt,
)


def test_direct_prompt_contains_source() -> None:
    source = "A difficult sentence."
    prompt = user_prompt(source, "direct")

    assert source in prompt
    assert prompt.endswith("Simplified sentence:")


def test_controlled_prompt_records_constraints() -> None:
    prompt = user_prompt(
        "A difficult sentence.",
        "controlled",
    )

    assert "Do not add information." in prompt
    assert "Preserve names, numbers, dates" in prompt
    assert prompt.endswith("Rewrite:")


def test_zero_shot_message_roles() -> None:
    messages = build_messages(
        "A difficult sentence.",
        "direct",
        0,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
    ]


def test_three_shot_message_roles() -> None:
    messages = build_messages(
        "A difficult sentence.",
        "controlled",
        3,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-2]["content"] == (
        FEW_SHOT_EXAMPLES[-1].target
    )


def test_fixed_few_shot_training_rows() -> None:
    assert FEW_SHOT_ROW_NUMBERS == (1, 2, 6)
    assert tuple(
        example.row_number
        for example in FEW_SHOT_EXAMPLES
    ) == FEW_SHOT_ROW_NUMBERS


def test_setting_names_round_trip() -> None:
    for setting in EXPERIMENT_SETTINGS:
        variant, shots = parse_setting(setting)

        assert setting_name(variant, shots) == setting


def test_prompt_record_includes_exact_examples() -> None:
    record = prompt_record("direct", 3)

    assert record["few_shot_row_numbers"] == [1, 2, 6]
    assert len(record["few_shot_examples"]) == 3


def test_invalid_prompt_options_are_rejected() -> None:
    with pytest.raises(ValueError):
        build_messages("A sentence.", "unknown", 0)

    with pytest.raises(ValueError):
        build_messages("A sentence.", "direct", 2)
