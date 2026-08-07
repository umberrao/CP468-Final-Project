from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch
from tqdm.auto import tqdm

from src.data import iter_jsonl
from src.inference import load_model, simplify_text
from src.metrics import (
    corpus_bleu,
    corpus_sari,
    sari_sentence,
)
from src.text import Vocabulary


EXPECTED_TEST_SIZE = 359
EXPECTED_REFERENCES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/raw/final/test.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/attention"),
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.test_data.exists():
        raise FileNotFoundError(args.test_data)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    if not args.vocabulary.exists():
        raise FileNotFoundError(args.vocabulary)

    records = list(iter_jsonl(args.test_data))

    if len(records) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SIZE} ASSET test "
            f"examples, found {len(records)}."
        )

    for index, record in enumerate(records):
        references = record.get("references", [])

        if len(references) != EXPECTED_REFERENCES:
            raise ValueError(
                f"Example {index} has {len(references)} "
                f"references; expected {EXPECTED_REFERENCES}."
            )

    if args.max_examples is not None:
        if args.max_examples < 1:
            raise ValueError(
                "--max-examples must be positive."
            )

        records = records[:args.max_examples]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocabulary = Vocabulary.load(args.vocabulary)

    model, config = load_model(
        args.checkpoint,
        vocabulary,
        device,
    )

    sources: list[str] = []
    predictions: list[str] = []
    all_references: list[list[str]] = []
    output_rows: list[dict[str, object]] = []

    start_time = time.perf_counter()

    for index, record in enumerate(
        tqdm(records, desc="Evaluating")
    ):
        source = record["source"]
        references = record["references"]

        prediction, _ = simplify_text(
            model,
            vocabulary,
            source,
            device,
            max_source_length=config["data"][
                "max_source_length"
            ],
            max_output_length=args.max_output_length,
        )

        example_sari = sari_sentence(
            source,
            prediction,
            references,
        )

        sources.append(source)
        predictions.append(prediction)
        all_references.append(references)

        output_rows.append(
            {
                "id": record.get(
                    "id",
                    record.get("gem_id", str(index)),
                ),
                "source": source,
                "target": record.get("target", ""),
                "prediction": prediction,
                "references": references,
                "sari": example_sari,
            }
        )

    elapsed = time.perf_counter() - start_time

    sari = corpus_sari(
        sources,
        predictions,
        all_references,
    )

    bleu = corpus_bleu(
        predictions,
        all_references,
    )

    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(0)
    else:
        hardware = "CPU"

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    predictions_path = (
        args.output_dir / "predictions.jsonl"
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in output_rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    use_attention = config["model"].get(
        "use_attention",
        True,
    )

    summary = {
        "model": (
            "lstm_attention"
            if use_attention
            else "lstm_no_attention"
        ),
        "use_attention": use_attention,
        "test_examples": len(records),
        "full_test_set": (
            len(records) == EXPECTED_TEST_SIZE
        ),
        "references_per_example": EXPECTED_REFERENCES,
        "sari": sari,
        "bleu": bleu,
        "evaluation_seconds": elapsed,
        "device": str(device),
        "hardware": hardware,
        "torch_version": torch.__version__,
        "sacrebleu_version": sacrebleu.__version__,
        "parameter_count": parameter_count,
        "max_output_length": args.max_output_length,
        "checkpoint": str(args.checkpoint),
    }

    metrics_path = args.output_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Examples: {len(records)}")
    print(f"SARI:     {sari:.4f}")
    print(f"BLEU:     {bleu:.4f}")
    print(f"Time:     {elapsed:.1f} seconds")
    print(f"Device:   {hardware}")
    print(f"Results:  {args.output_dir}")


if __name__ == "__main__":
    main()