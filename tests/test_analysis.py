import pytest

from src.analysis import (
    build_comparisons,
    count_error_flags,
    detect_error_flags,
    render_analysis_markdown,
    select_representative_examples,
)


def test_detects_generation_artifacts() -> None:
    flags = detect_error_flags(
        "A complex source sentence about a city.",
        "<unk> city city city < sep > city city city",
    )

    assert "unknown_token" in flags
    assert "separator_artifact" in flags
    assert "repetition" in flags


def test_detects_number_changes() -> None:
    flags = detect_error_flags(
        "The event happened in 1942.",
        "The event happened in 1950.",
    )

    assert "possible_number_loss" in flags
    assert "added_number" in flags


def test_detects_length_extremes() -> None:
    deletion_flags = detect_error_flags(
        "This is a considerably longer sentence with many facts.",
        "Facts vanished.",
    )
    copy_flags = detect_error_flags(
        "The city has a large public park.",
        "The city has a large public park.",
    )

    assert "possible_over_deletion" in deletion_flags
    assert "possible_under_simplification" in copy_flags


def make_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    attention = []
    no_attention = []
    qwen = []

    for index in range(12):
        source = (
            f"The complicated city event number {index} "
            "happened during the afternoon."
        )
        reference = (
            f"City event {index} happened in the afternoon."
        )
        shared = {
            "id": str(index),
            "source": source,
            "references": [reference],
        }

        attention.append(
            {
                **shared,
                "prediction": reference,
            }
        )
        no_attention.append(
            {
                **shared,
                "prediction": "The event happened.",
            }
        )
        qwen.append(
            {
                **shared,
                "prediction": (
                    f"The city event {index} happened "
                    "in the afternoon."
                ),
            }
        )

    return attention, no_attention, qwen


def test_builds_aligned_comparisons() -> None:
    attention, no_attention, qwen = make_rows()

    comparisons = build_comparisons(
        attention,
        no_attention,
        qwen,
    )

    assert len(comparisons) == 12
    assert set(comparisons[0]["predictions"]) == {
        "attention",
        "no_attention",
        "qwen_controlled_3shot",
    }


def test_rejects_misaligned_sources() -> None:
    attention, no_attention, qwen = make_rows()
    qwen[0] = {
        **qwen[0],
        "source": "A different source.",
    }

    with pytest.raises(ValueError):
        build_comparisons(
            attention,
            no_attention,
            qwen,
        )


def test_selects_ten_unique_examples() -> None:
    rows = build_comparisons(*make_rows())

    selected = select_representative_examples(rows)

    assert len(selected) == 10
    assert len(
        {example["index"] for example in selected}
    ) == 10
    assert all(
        "selection_reason" in example
        for example in selected
    )


def test_summarizes_and_renders_analysis() -> None:
    rows = build_comparisons(*make_rows())
    selected = select_representative_examples(rows)
    summary = count_error_flags(rows)

    markdown = render_analysis_markdown(
        selected,
        summary,
    )

    assert summary["examples"] == 12
    assert "Ten Representative Examples" in markdown
    assert markdown.count("### Example ") == 10
    assert "Manual assessment" in markdown