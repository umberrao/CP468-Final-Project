from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
from datasets import load_dataset


DATASET_NAME = "GEM/wiki_auto_asset_turk"


def clean_text(value: Any) -> str:
    """Normalize whitespace without changing the actual wording."""
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def source_key(text: str) -> str:
    """Create a stable key for leakage checks."""
    return clean_text(text).casefold()


def normalize_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    source = clean_text(raw.get("source"))
    target = clean_text(raw.get("target"))

    raw_references = raw.get("references") or []
    if isinstance(raw_references, str):
        raw_references = [raw_references]

    references = []
    seen_references = set()

    for reference in raw_references:
        reference = clean_text(reference)
        key = reference.casefold()

        if reference and key not in seen_references:
            references.append(reference)
            seen_references.add(key)

    if not references and target:
        references = [target]

    if not target and references:
        target = references[0]

    if not source or not target:
        return None

    example_id = clean_text(raw.get("gem_id"))
    if not example_id:
        value = f"{source}\n{target}".encode("utf-8")
        example_id = hashlib.sha256(value).hexdigest()[:16]

    return {
        "id": example_id,
        "source": source,
        "target": target,
        "references": references,
    }


def collect_split(
    split: str,
    limit: int | None,
    seed: int | None,
    shuffle_buffer: int,
    excluded_sources: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    if seed is not None:
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer,
        )

    rows = []
    seen_pairs = set()
    skipped_for_leakage = 0
    excluded_sources = excluded_sources or set()

    for raw in dataset:
        row = normalize_row(raw)

        if row is None:
            continue

        current_source_key = source_key(row["source"])

        if current_source_key in excluded_sources:
            skipped_for_leakage += 1
            continue

        pair_key = (
            current_source_key,
            row["target"].casefold(),
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        rows.append(row)

        if limit is not None and len(rows) >= limit:
            break

    if limit is not None and len(rows) < limit:
        raise RuntimeError(
            f"Requested {limit} examples from {split}, "
            f"but only collected {len(rows)}."
        )

    return rows, skipped_for_leakage


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-free WikiAuto and ASSET splits."
    )
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=468)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # These held-out splits are collected before training data.
    validation_rows, _ = collect_split(
        split="validation",
        limit=args.validation_size,
        seed=args.seed + 1,
        shuffle_buffer=args.shuffle_buffer,
    )

    test_rows, _ = collect_split(
        split="test_asset",
        limit=None,
        seed=None,
        shuffle_buffer=args.shuffle_buffer,
    )

    validation_sources = {
        source_key(row["source"]) for row in validation_rows
    }
    test_sources = {
        source_key(row["source"]) for row in test_rows
    }

    validation_test_overlap = validation_sources & test_sources
    if validation_test_overlap:
        raise RuntimeError(
            "Validation and test source sentences overlap."
        )

    held_out_sources = validation_sources | test_sources

    train_rows, removed_train_overlaps = collect_split(
        split="train",
        limit=args.train_size,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        excluded_sources=held_out_sources,
    )

    train_sources = {
        source_key(row["source"]) for row in train_rows
    }

    if train_sources & held_out_sources:
        raise RuntimeError("Training leakage was detected.")

    paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
        "test": args.output_dir / "test.jsonl",
    }

    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["validation"], validation_rows)
    write_jsonl(paths["test"], test_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "datasets_library_version": datasets.__version__,
        "seed": args.seed,
        "licenses": {
            "WikiAuto_train_validation": "CC BY-NC 3.0",
            "ASSET_test": "CC BY-NC 4.0",
        },
        "splits": {
            name: {
                "examples": len(rows),
                "file": str(paths[name]),
                "sha256": file_sha256(paths[name]),
            }
            for name, rows in {
                "train": train_rows,
                "validation": validation_rows,
                "test": test_rows,
            }.items()
        },
        "leakage_check": {
            "train_holdout_overlap": 0,
            "validation_test_overlap": 0,
            "removed_train_rows": removed_train_overlaps,
        },
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Train examples:      {len(train_rows)}")
    print(f"Validation examples: {len(validation_rows)}")
    print(f"Test examples:       {len(test_rows)}")
    print(f"Removed overlaps:    {removed_train_overlaps}")
    print("Leakage check:       PASSED")
    print(f"Manifest:            {manifest_path}")


if __name__ == "__main__":
    main()