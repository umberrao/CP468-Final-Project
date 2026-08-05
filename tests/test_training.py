import random

import numpy as np
import torch
from torch.nn import functional as F

from src.models.seq2seq import Seq2Seq
from src.training import (
    count_trainable_parameters,
    run_epoch,
    sequence_cross_entropy,
    set_seed,
)


def create_model() -> Seq2Seq:
    return Seq2Seq(
        vocab_size=20,
        embedding_dim=8,
        encoder_hidden_dim=10,
        decoder_hidden_dim=12,
        attention_dim=10,
        pad_id=0,
        dropout=0.0,
    )


def create_batch() -> dict:
    source_ids = torch.tensor(
        [
            [2, 4, 5, 3],
            [2, 6, 3, 0],
        ]
    )

    return {
        "source_ids": source_ids,
        "source_lengths": torch.tensor([4, 3]),
        "source_mask": source_ids.ne(0),
        "target_input_ids": torch.tensor(
            [
                [2, 8, 9],
                [2, 10, 0],
            ]
        ),
        "target_output_ids": torch.tensor(
            [
                [8, 9, 3],
                [10, 3, 0],
            ]
        ),
    }


def test_cross_entropy_ignores_padding() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 0.0],
                [100.0, -100.0, 50.0],
            ]
        ]
    )
    targets = torch.tensor([[1, 0]])

    actual = sequence_cross_entropy(
        logits,
        targets,
        pad_id=0,
    )
    expected = F.cross_entropy(
        logits[:, 0],
        torch.tensor([1]),
    )

    assert torch.allclose(actual, expected)


def test_set_seed_is_reproducible() -> None:
    set_seed(468)

    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    set_seed(468)

    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_training_epoch_updates_model() -> None:
    set_seed(468)

    model = create_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )
    batch = create_batch()

    before = (
        model.decoder.output_projection.weight
        .detach()
        .clone()
    )

    result = run_epoch(
        model,
        [batch],
        pad_id=0,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    after = (
        model.decoder.output_projection.weight
        .detach()
        .clone()
    )

    assert result.loss > 0
    assert result.perplexity > 1
    assert result.token_count == 5
    assert not torch.equal(before, after)
    assert count_trainable_parameters(model) > 0