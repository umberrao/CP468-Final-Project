from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.seq2seq import Seq2Seq
from src.text import Vocabulary


def load_model(
    checkpoint_path: str | Path,
    vocabulary: Vocabulary,
    device: torch.device,
) -> tuple[Seq2Seq, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint["config"]

    model = Seq2Seq(
        vocab_size=len(vocabulary),
        embedding_dim=config["model"]["embedding_dim"],
        encoder_hidden_dim=config["model"][
            "encoder_hidden_dim"
        ],
        decoder_hidden_dim=config["model"][
            "decoder_hidden_dim"
        ],
        attention_dim=config["model"]["attention_dim"],
        pad_id=vocabulary.pad_id,
        dropout=config["model"]["dropout"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, config


@torch.no_grad()
def simplify_text(
    model: Seq2Seq,
    vocabulary: Vocabulary,
    text: str,
    device: torch.device,
    max_source_length: int = 80,
    max_output_length: int = 80,
) -> tuple[str, torch.Tensor]:
    encoded = vocabulary.encode(
        text,
        max_length=max_source_length,
    )

    source_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device,
    )
    source_lengths = torch.tensor(
        [len(encoded)],
        dtype=torch.long,
        device=device,
    )
    source_mask = source_ids.ne(vocabulary.pad_id)

    generated, attention = model.greedy_decode(
        source_ids,
        source_lengths,
        source_mask,
        bos_id=vocabulary.bos_id,
        eos_id=vocabulary.eos_id,
        max_length=max_output_length,
    )

    simplified = vocabulary.decode(
        generated[0].tolist()
    )

    return simplified, attention.cpu()