from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.analysis import (
    FLAG_LABELS,
    MODEL_KEYS,
    build_comparisons,
    count_error_flags,
    prediction_text,
    render_analysis_markdown,
    select_representative_examples,
)


EXPECTED_TEST_SIZE = 359


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare attention, ablation, and Qwen predictions."
        )
    )
    parser.add_argument(
        "--attention",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--no-attention",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--exclude-index",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Zero-based test indices to omit from the ten "
            "qualitative examples."
        ),
    )

    return parser.parse_args()


def load_prediction_rows(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line "
                    f"{line_number}."
                ) from error

            if not isinstance(row.get("source"), str):
                raise ValueError(
                    f"Missing source in {path} on line "
                    f"{line_number}."
                )

            prediction_text(row)
            rows.append(row)

    if len(rows) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SIZE} rows in {path}, "
            f"found {len(rows)}."
        )

    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_error_csv(
    path: Path,
    error_summary: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        header = ["flag"]

        for model in MODEL_KEYS:
            header.extend(
                [
                    f"{model}_count",
                    f"{model}_percentage",
                ]
            )

        writer.writerow(header)

        for flag in FLAG_LABELS:
            row: list[Any] = [flag]

            for model in MODEL_KEYS:
                result = error_summary["models"][model][
                    "flags"
                ][flag]
                row.extend(
                    [
                        result["count"],
                        round(result["percentage"], 4),
                    ]
                )

            writer.writerow(row)


def main() -> None:
    args = parse_args()

    if args.example_count < 10:
        raise ValueError(
            "--example-count must be at least 10."
        )

    attention_rows = load_prediction_rows(
        args.attention
    )
    no_attention_rows = load_prediction_rows(
        args.no_attention
    )
    qwen_rows = load_prediction_rows(args.qwen)

    comparisons = build_comparisons(
        attention_rows,
        no_attention_rows,
        qwen_rows,
    )
    error_summary = count_error_flags(comparisons)

    excluded_indices = set(args.exclude_index)
    selection_pool = [
        comparison
        for comparison in comparisons
        if comparison["index"] not in excluded_indices
    ]

    selected = select_representative_examples(
        selection_pool,
        count=args.example_count,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_summary = {
        "test_examples": len(comparisons),
        "references_per_example": len(
            attention_rows[0]["references"]
        ),
        "models": {
            "no_attention": str(args.no_attention),
            "attention": str(args.attention),
            "qwen_controlled_3shot": str(args.qwen),
        },
        "selection_method": [
            "three largest attention advantages over ablation",
            "three largest Qwen advantages over attention",
            "two lowest attention SARI examples",
            "two lowest Qwen SARI examples",
        ],
        "excluded_indices": sorted(excluded_indices),
        "error_summary": error_summary,
        "selected_examples": selected,
    }

    summary_path = (
        args.output_dir / "analysis_summary.json"
    )
    markdown_path = (
        args.output_dir / "qualitative_examples.md"
    )
    csv_path = args.output_dir / "error_counts.csv"

    write_json(summary_path, analysis_summary)
    markdown_path.write_text(
        render_analysis_markdown(
            selected,
            error_summary,
        ),
        encoding="utf-8",
    )
    write_error_csv(csv_path, error_summary)

    print(f"Aligned examples:  {len(comparisons)}")
    print(f"Selected examples: {len(selected)}")
    print(f"Summary:           {summary_path}")
    print(f"Markdown:          {markdown_path}")
    print(f"Error counts:      {csv_path}")


if __name__ == "__main__":
    main()