from pathlib import Path

from scripts.create_figures import (
    build_score_rows,
    create_all_figures,
    load_metrics,
)


def test_builds_all_six_score_rows() -> None:
    metrics = load_metrics(
        Path("results/final_metrics.json")
    )

    rows = build_score_rows(metrics)

    assert len(rows) == 6
    assert rows[0]["label"] == (
        "LSTM without attention"
    )
    assert rows[-1]["label"] == (
        "Qwen controlled 3-shot"
    )
    assert rows[-1]["sari"] == 49.7276


def test_creates_three_nonempty_figures(
    tmp_path: Path,
) -> None:
    paths = create_all_figures(
        Path("results/final_metrics.json"),
        Path("results/analysis/error_counts.csv"),
        tmp_path,
    )

    assert {path.name for path in paths} == {
        "model_scores.png",
        "qwen_prompt_comparison.png",
        "error_rates.png",
    }

    assert all(path.exists() for path in paths)
    assert all(
        path.stat().st_size > 10_000
        for path in paths
    )