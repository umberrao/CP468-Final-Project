from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SARI_COLOR = "#356A9A"
BLEU_COLOR = "#D68A33"
MODEL_COLORS = ("#9B6A9E", "#356A9A", "#3E8E7E")

ERROR_FLAGS = (
    ("unknown_token", "Unknown tokens"),
    ("repetition", "Repetition"),
    ("possible_over_deletion", "Over-deletion"),
    ("possible_under_simplification", "Under-simplification"),
    ("possible_number_loss", "Number loss"),
    ("possible_name_loss", "Name loss"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final project figures."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/final_metrics.json"),
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("results/analysis/error_counts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/figures"),
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def load_error_rows(
    path: Path,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    return {
        row["flag"]: row
        for row in rows
    }


def build_score_rows(
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    custom = metrics["custom_models"]
    settings = metrics["llm"]["settings"]

    return [
        {
            "label": "LSTM without attention",
            "sari": custom["no_attention"]["sari"],
            "bleu": custom["no_attention"]["bleu"],
        },
        {
            "label": "LSTM with attention",
            "sari": custom["attention"]["sari"],
            "bleu": custom["attention"]["bleu"],
        },
        {
            "label": "Qwen direct 0-shot",
            "sari": settings["direct_0shot"]["sari"],
            "bleu": settings["direct_0shot"]["bleu"],
        },
        {
            "label": "Qwen controlled 0-shot",
            "sari": settings["controlled_0shot"]["sari"],
            "bleu": settings["controlled_0shot"]["bleu"],
        },
        {
            "label": "Qwen direct 3-shot",
            "sari": settings["direct_3shot"]["sari"],
            "bleu": settings["direct_3shot"]["bleu"],
        },
        {
            "label": "Qwen controlled 3-shot",
            "sari": settings["controlled_3shot"]["sari"],
            "bleu": settings["controlled_3shot"]["bleu"],
        },
    ]


def style_axes(
    ax: Any,
    grid_axis: str,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis=grid_axis,
        color="#D7DCE2",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.set_axisbelow(True)


def save_figure(
    fig: Any,
    path: Path,
) -> None:
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_model_scores(
    metrics: dict[str, Any],
    path: Path,
) -> None:
    rows = build_score_rows(metrics)
    labels = [row["label"] for row in rows]
    sari = [row["sari"] for row in rows]
    bleu = [row["bleu"] for row in rows]

    positions = np.arange(len(rows))
    height = 0.34

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2),
    )

    sari_bars = ax.barh(
        positions - height / 2,
        sari,
        height,
        label="SARI",
        color=SARI_COLOR,
    )
    bleu_bars = ax.barh(
        positions + height / 2,
        bleu,
        height,
        label="BLEU",
        color=BLEU_COLOR,
    )

    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (higher is better)")
    ax.set_title(
        "Text simplification performance on ASSET\n"
        "359 examples with 10 references each",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        loc="lower right",
        frameon=False,
        ncols=2,
    )
    ax.bar_label(
        sari_bars,
        fmt="%.1f",
        padding=3,
        fontsize=8,
    )
    ax.bar_label(
        bleu_bars,
        fmt="%.1f",
        padding=3,
        fontsize=8,
    )

    style_axes(ax, "x")
    fig.tight_layout()
    save_figure(fig, path)


def plot_qwen_prompts(
    metrics: dict[str, Any],
    path: Path,
) -> None:
    settings = metrics["llm"]["settings"]

    keys = (
        "direct_0shot",
        "controlled_0shot",
        "direct_3shot",
        "controlled_3shot",
    )
    labels = (
        "Direct\n0-shot",
        "Controlled\n0-shot",
        "Direct\n3-shot",
        "Controlled\n3-shot",
    )

    sari = [
        settings[key]["sari"]
        for key in keys
    ]
    bleu = [
        settings[key]["bleu"]
        for key in keys
    ]

    positions = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9.2, 5.8),
    )

    sari_bars = ax.bar(
        positions - width / 2,
        sari,
        width,
        label="SARI",
        color=SARI_COLOR,
    )
    bleu_bars = ax.bar(
        positions + width / 2,
        bleu,
        width,
        label="BLEU",
        color=BLEU_COLOR,
    )

    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(
        "Effect of Qwen prompt design\n"
        "Controlled three-shot prompting performed best",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        loc="upper left",
        frameon=False,
        ncols=2,
    )
    ax.bar_label(
        sari_bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )
    ax.bar_label(
        bleu_bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

    style_axes(ax, "y")
    fig.tight_layout()
    save_figure(fig, path)


def plot_error_rates(
    error_rows: dict[str, dict[str, str]],
    path: Path,
) -> None:
    labels = [
        label
        for _, label in ERROR_FLAGS
    ]
    no_attention = [
        float(
            error_rows[flag][
                "no_attention_percentage"
            ]
        )
        for flag, _ in ERROR_FLAGS
    ]
    attention = [
        float(
            error_rows[flag][
                "attention_percentage"
            ]
        )
        for flag, _ in ERROR_FLAGS
    ]
    qwen = [
        float(
            error_rows[flag][
                "qwen_controlled_3shot_percentage"
            ]
        )
        for flag, _ in ERROR_FLAGS
    ]

    positions = np.arange(len(ERROR_FLAGS))
    height = 0.23

    fig, ax = plt.subplots(
        figsize=(10.8, 6.4),
    )

    no_attention_bars = ax.barh(
        positions - height,
        no_attention,
        height,
        label="LSTM without attention",
        color=MODEL_COLORS[0],
    )
    attention_bars = ax.barh(
        positions,
        attention,
        height,
        label="LSTM with attention",
        color=MODEL_COLORS[1],
    )
    qwen_bars = ax.barh(
        positions + height,
        qwen,
        height,
        label="Qwen controlled three-shot",
        color=MODEL_COLORS[2],
    )

    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Examples flagged (%)")
    ax.set_title(
        "Automatic diagnostic flags by model\n"
        "Flags identify outputs requiring manual review",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.99, 0.43),
        frameon=False,
    )

    for bars in (
        no_attention_bars,
        attention_bars,
        qwen_bars,
    ):
        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=3,
            fontsize=7,
        )

    style_axes(ax, "x")
    fig.tight_layout()
    save_figure(fig, path)


def create_all_figures(
    metrics_path: Path,
    errors_path: Path,
    output_dir: Path,
) -> list[Path]:
    metrics = load_metrics(metrics_path)
    error_rows = load_error_rows(errors_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        output_dir / "model_scores.png",
        output_dir / "qwen_prompt_comparison.png",
        output_dir / "error_rates.png",
    ]

    plot_model_scores(
        metrics,
        paths[0],
    )
    plot_qwen_prompts(
        metrics,
        paths[1],
    )
    plot_error_rates(
        error_rows,
        paths[2],
    )

    return paths


def main() -> None:
    args = parse_args()

    paths = create_all_figures(
        args.metrics,
        args.errors,
        args.output_dir,
    )

    for path in paths:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()