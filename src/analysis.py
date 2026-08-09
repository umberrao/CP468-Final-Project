from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

from src.metrics import sari_sentence


MODEL_KEYS = (
    "no_attention",
    "attention",
    "qwen_controlled_3shot",
)

MODEL_LABELS = {
    "no_attention": "LSTM without attention",
    "attention": "LSTM with attention",
    "qwen_controlled_3shot": "Qwen controlled three-shot",
}

FLAG_LABELS = {
    "empty_output": "Empty output",
    "unknown_token": "Unknown-token artifact",
    "separator_artifact": "Separator artifact",
    "repetition": "Repeated word or phrase",
    "possible_over_deletion": "Possible over-deletion",
    "possible_under_simplification": (
        "Possible under-simplification"
    ),
    "possible_number_loss": "Possible number loss",
    "added_number": "Number not found in source",
    "possible_name_loss": "Possible name loss",
}

_WORD_PATTERN = re.compile(
    r"\b[\w]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][\w]+)*\b",
    flags=re.UNICODE,
)
_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)*\b"
)
_CAPITALIZED_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z\N{LATIN CAPITAL LETTER A WITH GRAVE}-"
    r"\N{LATIN SMALL LETTER Y WITH DIAERESIS}'"
    r"\N{RIGHT SINGLE QUOTATION MARK}-]*\b"
)
_UNKNOWN_PATTERN = re.compile(
    r"<\s*unk\s*>",
    flags=re.IGNORECASE,
)
_SEPARATOR_PATTERN = re.compile(
    r"<\s*sep\s*>",
    flags=re.IGNORECASE,
)

_COMMON_CAPITALIZED_WORDS = {
    "a",
    "after",
    "although",
    "an",
    "as",
    "at",
    "before",
    "during",
    "for",
    "from",
    "he",
    "her",
    "his",
    "however",
    "i",
    "in",
    "it",
    "its",
    "on",
    "she",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "those",
    "to",
    "we",
    "when",
    "while",
    "with",
}


def word_tokens(text: str) -> list[str]:
    return [
        match.group(0).casefold()
        for match in _WORD_PATTERN.finditer(text)
    ]


def number_tokens(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _NUMBER_PATTERN.finditer(text)
    }


def capitalized_tokens(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _CAPITALIZED_PATTERN.finditer(text)
        if match.group(0).casefold()
        not in _COMMON_CAPITALIZED_WORDS
    }


def contains_repetition(tokens: Sequence[str]) -> bool:
    if any(
        first == second
        for first, second in zip(tokens, tokens[1:])
    ):
        return True

    for order in (2, 3):
        ngrams = [
            tuple(tokens[index:index + order])
            for index in range(
                len(tokens) - order + 1
            )
        ]

        if any(
            count >= 3
            for count in Counter(ngrams).values()
        ):
            return True

    return False


def detect_error_flags(
    source: str,
    prediction: str,
) -> list[str]:
    """Return deterministic flags for outputs needing review."""
    source_words = word_tokens(source)
    prediction_words = word_tokens(prediction)
    flags = []

    if not prediction_words:
        return ["empty_output"]

    if _UNKNOWN_PATTERN.search(prediction):
        flags.append("unknown_token")

    if _SEPARATOR_PATTERN.search(prediction):
        flags.append("separator_artifact")

    if contains_repetition(prediction_words):
        flags.append("repetition")

    length_ratio = (
        len(prediction_words) / len(source_words)
        if source_words
        else 1.0
    )

    if (
        len(source_words) >= 8
        and length_ratio < 0.35
    ):
        flags.append("possible_over_deletion")

    similarity = SequenceMatcher(
        None,
        source_words,
        prediction_words,
    ).ratio()

    if length_ratio >= 0.9 and similarity >= 0.92:
        flags.append("possible_under_simplification")

    source_numbers = number_tokens(source)
    prediction_numbers = number_tokens(prediction)

    if source_numbers - prediction_numbers:
        flags.append("possible_number_loss")

    if prediction_numbers - source_numbers:
        flags.append("added_number")

    source_names = capitalized_tokens(source)
    prediction_name_words = set(prediction_words)

    if source_names - prediction_name_words:
        flags.append("possible_name_loss")

    return flags


def prediction_text(row: dict[str, Any]) -> str:
    for key in (
        "prediction",
        "output",
        "simplification",
    ):
        value = row.get(key)

        if isinstance(value, str):
            return value.strip()

    raise ValueError(
        "Prediction row has no prediction, output, "
        "or simplification field."
    )


def build_comparisons(
    attention_rows: Sequence[dict[str, Any]],
    no_attention_rows: Sequence[dict[str, Any]],
    qwen_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lengths = {
        len(attention_rows),
        len(no_attention_rows),
        len(qwen_rows),
    }

    if len(lengths) != 1:
        raise ValueError(
            "All prediction files must contain the same "
            "number of examples."
        )

    if not attention_rows:
        raise ValueError("Prediction files cannot be empty.")

    comparisons = []

    for index, (
        attention_row,
        no_attention_row,
        qwen_row,
    ) in enumerate(
        zip(
            attention_rows,
            no_attention_rows,
            qwen_rows,
        )
    ):
        source = str(attention_row.get("source", ""))

        if not source:
            raise ValueError(
                f"Example {index} has no source text."
            )

        other_sources = {
            str(no_attention_row.get("source", "")),
            str(qwen_row.get("source", "")),
        }

        if other_sources != {source}:
            raise ValueError(
                "Prediction files are not aligned at "
                f"example {index}."
            )

        references = attention_row.get(
            "references",
            [],
        )

        if not isinstance(references, list) or not references:
            raise ValueError(
                f"Example {index} has no references."
            )

        for row in (no_attention_row, qwen_row):
            other_references = row.get("references")

            if (
                other_references is not None
                and other_references != references
            ):
                raise ValueError(
                    "Reference sets are not aligned at "
                    f"example {index}."
                )

        predictions = {
            "no_attention": prediction_text(
                no_attention_row
            ),
            "attention": prediction_text(
                attention_row
            ),
            "qwen_controlled_3shot": prediction_text(
                qwen_row
            ),
        }

        scores = {
            model: sari_sentence(
                source,
                prediction,
                references,
            )
            for model, prediction in predictions.items()
        }

        flags = {
            model: detect_error_flags(
                source,
                prediction,
            )
            for model, prediction in predictions.items()
        }

        comparisons.append(
            {
                "index": index,
                "id": attention_row.get(
                    "id",
                    attention_row.get(
                        "gem_id",
                        str(index),
                    ),
                ),
                "source": source,
                "human_reference": references[0],
                "predictions": predictions,
                "sari": scores,
                "flags": flags,
            }
        )

    return comparisons


def count_error_flags(
    comparisons: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not comparisons:
        raise ValueError("Comparisons cannot be empty.")

    model_summaries = {}

    for model in MODEL_KEYS:
        counter = Counter(
            flag
            for comparison in comparisons
            for flag in comparison["flags"][model]
        )
        flagged_examples = sum(
            bool(comparison["flags"][model])
            for comparison in comparisons
        )
        average_sari = sum(
            comparison["sari"][model]
            for comparison in comparisons
        ) / len(comparisons)

        model_summaries[model] = {
            "average_sentence_sari": average_sari,
            "flagged_examples": flagged_examples,
            "flagged_percentage": (
                100.0
                * flagged_examples
                / len(comparisons)
            ),
            "flags": {
                flag: {
                    "count": counter[flag],
                    "percentage": (
                        100.0
                        * counter[flag]
                        / len(comparisons)
                    ),
                }
                for flag in FLAG_LABELS
            },
        }

    return {
        "examples": len(comparisons),
        "note": (
            "Flags are deterministic indicators for manual "
            "review, not definitive factual-error labels."
        ),
        "models": model_summaries,
    }


def _add_from_pool(
    selected: list[dict[str, Any]],
    selected_indices: set[int],
    pool: Sequence[dict[str, Any]],
    amount: int,
    reason: str,
) -> None:
    added = 0

    for comparison in pool:
        index = int(comparison["index"])

        if index in selected_indices:
            continue

        selected_example = dict(comparison)
        selected_example["selection_reason"] = reason
        selected.append(selected_example)
        selected_indices.add(index)
        added += 1

        if added >= amount:
            break


def select_representative_examples(
    comparisons: Sequence[dict[str, Any]],
    count: int = 10,
) -> list[dict[str, Any]]:
    if count < 10:
        raise ValueError(
            "At least 10 examples are required."
        )

    if len(comparisons) < count:
        raise ValueError(
            "Not enough comparisons for the requested count."
        )

    attention_advantage = sorted(
        comparisons,
        key=lambda row: (
            row["sari"]["attention"]
            - row["sari"]["no_attention"]
        ),
        reverse=True,
    )
    qwen_advantage = sorted(
        comparisons,
        key=lambda row: (
            row["sari"]["qwen_controlled_3shot"]
            - row["sari"]["attention"]
        ),
        reverse=True,
    )
    attention_failures = sorted(
        comparisons,
        key=lambda row: row["sari"]["attention"],
    )
    qwen_failures = sorted(
        comparisons,
        key=lambda row: (
            row["sari"]["qwen_controlled_3shot"]
        ),
    )

    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()

    _add_from_pool(
        selected,
        selected_indices,
        attention_advantage,
        3,
        "Large attention advantage over the ablation",
    )
    _add_from_pool(
        selected,
        selected_indices,
        qwen_advantage,
        3,
        "Large Qwen advantage over the attention model",
    )
    _add_from_pool(
        selected,
        selected_indices,
        attention_failures,
        2,
        "Low attention-model SARI",
    )
    _add_from_pool(
        selected,
        selected_indices,
        qwen_failures,
        2,
        "Low Qwen SARI",
    )

    if len(selected) < count:
        score_spread = sorted(
            comparisons,
            key=lambda row: (
                max(row["sari"].values())
                - min(row["sari"].values())
            ),
            reverse=True,
        )
        _add_from_pool(
            selected,
            selected_indices,
            score_spread,
            count - len(selected),
            "Large difference between systems",
        )

    return selected[:count]


def _markdown_text(value: Any) -> str:
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
        .strip()
    )


def _format_flags(flags: Sequence[str]) -> str:
    if not flags:
        return "None"

    return ", ".join(
        FLAG_LABELS[flag]
        for flag in flags
    )


def render_analysis_markdown(
    selected: Sequence[dict[str, Any]],
    error_summary: dict[str, Any],
) -> str:
    lines = [
        "# Qualitative and Error Analysis",
        "",
        (
            "All systems use the same ASSET source sentences and "
            "human references. SARI is shown per example."
        ),
        "",
        "## Automatic Diagnostic Flags",
        "",
        (
            "These are reproducible indicators for manual review. "
            "They are not treated as definitive factual errors."
        ),
        "",
        (
            "| Flag | LSTM without attention | "
            "LSTM with attention | Qwen controlled three-shot |"
        ),
        "| --- | ---: | ---: | ---: |",
    ]

    for flag, label in FLAG_LABELS.items():
        values = []

        for model in MODEL_KEYS:
            result = error_summary["models"][model][
                "flags"
            ][flag]
            values.append(
                f"{result['count']} "
                f"({result['percentage']:.1f}%)"
            )

        lines.append(
            f"| {label} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Ten Representative Examples",
            "",
        ]
    )

    for number, example in enumerate(selected, start=1):
        lines.extend(
            [
                (
                    f"### Example {number}: "
                    f"{example['selection_reason']}"
                ),
                "",
                (
                    "**Source:** "
                    f"{_markdown_text(example['source'])}"
                ),
                "",
                (
                    "**Human reference:** "
                    f"{_markdown_text(example['human_reference'])}"
                ),
                "",
                "| System | Output | SARI | Automatic flags |",
                "| --- | --- | ---: | --- |",
            ]
        )

        for model in MODEL_KEYS:
            lines.append(
                f"| {MODEL_LABELS[model]} | "
                f"{_markdown_text(example['predictions'][model])} | "
                f"{example['sari'][model]:.2f} | "
                f"{_format_flags(example['flags'][model])} |"
            )

        lines.extend(
            [
                "",
                (
                    "**Manual assessment:** TODO — identify "
                    "successful simplification, meaning loss, "
                    "unsupported addition, repetition, fluency "
                    "problem, or insufficient simplification."
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"