from __future__ import annotations

import torch
from torch import Tensor, nn


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention over encoder outputs."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int,
    ) -> None:
        super().__init__()

        self.encoder_projection = nn.Linear(
            encoder_dim,
            attention_dim,
            bias=False,
        )
        self.decoder_projection = nn.Linear(
            decoder_dim,
            attention_dim,
            bias=False,
        )
        self.score_projection = nn.Linear(
            attention_dim,
            1,
            bias=False,
        )

    def forward(
        self,
        decoder_hidden: Tensor,
        encoder_outputs: Tensor,
        source_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if source_mask.shape != encoder_outputs.shape[:2]:
            raise ValueError(
                "source_mask must match the encoder batch "
                "and sequence dimensions."
            )

        source_mask = source_mask.bool()

        projected_encoder = self.encoder_projection(
            encoder_outputs
        )
        projected_decoder = self.decoder_projection(
            decoder_hidden
        ).unsqueeze(1)

        energy = torch.tanh(
            projected_encoder + projected_decoder
        )
        scores = self.score_projection(energy).squeeze(-1)

        scores = scores.masked_fill(
            ~source_mask,
            torch.finfo(scores.dtype).min,
        )

        attention_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context, attention_weights