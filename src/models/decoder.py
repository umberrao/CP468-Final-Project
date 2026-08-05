from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.attention import AdditiveAttention


class Decoder(nn.Module):
    """One decoding step with LSTM and additive attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        pad_id: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.attention = AdditiveAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
        )

        self.lstm_cell = nn.LSTMCell(
            input_size=embedding_dim + encoder_dim,
            hidden_size=decoder_hidden_dim,
        )

        self.output_projection = nn.Linear(
            decoder_hidden_dim + encoder_dim + embedding_dim,
            vocab_size,
        )

    def forward(
        self,
        input_ids: Tensor,
        hidden: Tensor,
        cell: Tensor,
        encoder_outputs: Tensor,
        source_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        embedded = self.embedding_dropout(
            self.embedding(input_ids)
        )

        context, attention_weights = self.attention(
            hidden,
            encoder_outputs,
            source_mask,
        )

        lstm_input = torch.cat(
            (embedded, context),
            dim=1,
        )

        next_hidden, next_cell = self.lstm_cell(
            lstm_input,
            (hidden, cell),
        )

        output_features = torch.cat(
            (next_hidden, context, embedded),
            dim=1,
        )
        logits = self.output_projection(output_features)

        return (
            logits,
            next_hidden,
            next_cell,
            attention_weights,
        )