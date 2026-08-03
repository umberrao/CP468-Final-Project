import json

import torch

from src.data import (
    SimplificationDataset,
    build_vocabulary,
    collate_batch,
)


def create_sample_file(tmp_path):
    records = [
        {
            "id": "one",
            "source": "A very complicated sentence.",
            "target": "A simple sentence.",
            "references": ["A simple sentence."],
        },
        {
            "id": "two",
            "source": "Short text.",
            "target": "Short.",
            "references": ["Short."],
        },
    ]

    path = tmp_path / "sample.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_build_vocabulary_from_training_only(tmp_path) -> None:
    path = create_sample_file(tmp_path)

    vocabulary = build_vocabulary(
        path,
        min_frequency=1,
        max_size=100,
    )

    assert "sentence" in vocabulary.token_to_id
    assert (
        vocabulary.encode(
            "unseen",
            add_special_tokens=False,
        )[0]
        == vocabulary.unk_id
    )


def test_dataset_adds_bos_and_eos(tmp_path) -> None:
    path = create_sample_file(tmp_path)
    vocabulary = build_vocabulary(path, min_frequency=1)
    dataset = SimplificationDataset(path, vocabulary)

    example = dataset[0]

    assert example["source_ids"][0] == vocabulary.bos_id
    assert example["source_ids"][-1] == vocabulary.eos_id
    assert example["target_ids"][0] == vocabulary.bos_id
    assert example["target_ids"][-1] == vocabulary.eos_id


def test_collate_padding_and_masks(tmp_path) -> None:
    path = create_sample_file(tmp_path)
    vocabulary = build_vocabulary(path, min_frequency=1)
    dataset = SimplificationDataset(path, vocabulary)

    batch = collate_batch(
        [dataset[0], dataset[1]],
        vocabulary.pad_id,
    )

    assert batch["source_ids"].shape[0] == 2
    assert batch["source_mask"].dtype == torch.bool

    assert torch.equal(
        batch["source_mask"].sum(dim=1),
        batch["source_lengths"],
    )

    assert (
        batch["target_input_ids"].shape
        == batch["target_output_ids"].shape
    )

    assert batch["target_mask"].dtype == torch.bool