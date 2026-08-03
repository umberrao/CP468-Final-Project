from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.text import Vocabulary


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if "source" not in record or "target" not in record:
                raise ValueError(
                    f"Missing source or target on line {line_number}."
                )

            yield record


def build_vocabulary(
    train_path: str | Path,
    min_frequency: int = 2,
    max_size: int = 20_000,
) -> Vocabulary:
    """Build vocabulary exclusively from the training split."""

    def training_texts() -> Iterator[str]:
        for record in iter_jsonl(train_path):
            yield record["source"]
            yield record["target"]

    return Vocabulary.build(
        training_texts(),
        min_frequency=min_frequency,
        max_size=max_size,
    )


class SimplificationDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        vocabulary: Vocabulary,
        max_source_length: int = 80,
        max_target_length: int = 80,
    ) -> None:
        self.records = list(iter_jsonl(path))
        self.vocabulary = vocabulary
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        source_ids = self.vocabulary.encode(
            record["source"],
            max_length=self.max_source_length,
        )
        target_ids = self.vocabulary.encode(
            record["target"],
            max_length=self.max_target_length,
        )

        return {
            "id": record.get("id", str(index)),
            "source_text": record["source"],
            "target_text": record["target"],
            "references": record.get(
                "references",
                [record["target"]],
            ),
            "source_ids": torch.tensor(
                source_ids,
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(
                target_ids,
                dtype=torch.long,
            ),
        }


def collate_batch(
    batch: list[dict[str, Any]],
    pad_id: int,
) -> dict[str, Any]:
    source_sequences = [
        example["source_ids"] for example in batch
    ]
    target_sequences = [
        example["target_ids"] for example in batch
    ]

    source_lengths = torch.tensor(
        [len(sequence) for sequence in source_sequences],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [len(sequence) for sequence in target_sequences],
        dtype=torch.long,
    )

    source_ids = pad_sequence(
        source_sequences,
        batch_first=True,
        padding_value=pad_id,
    )
    target_ids = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=pad_id,
    )

    return {
        "ids": [example["id"] for example in batch],
        "source_texts": [
            example["source_text"] for example in batch
        ],
        "target_texts": [
            example["target_text"] for example in batch
        ],
        "references": [
            example["references"] for example in batch
        ],
        "source_ids": source_ids,
        "source_lengths": source_lengths,
        "source_mask": source_ids.ne(pad_id),
        "target_ids": target_ids,
        "target_lengths": target_lengths,
        "target_input_ids": target_ids[:, :-1],
        "target_output_ids": target_ids[:, 1:],
        "target_mask": target_ids[:, 1:].ne(pad_id),
    }