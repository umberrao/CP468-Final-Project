from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.attention import AdditiveAttention


class Decoder(nn.Module):
    """One decoding step with optional additive attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        pad_id: int,
        dropout: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        self.use_attention = use_attention

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        if use_attention:
            self.attention: AdditiveAttention | None = (
                AdditiveAttention(
                    encoder_dim=encoder_dim,
                    decoder_dim=decoder_hidden_dim,
                    attention_dim=attention_dim,
                )
            )
        else:
            self.attention = None

        lstm_input_size = embedding_dim

        if use_attention:
            lstm_input_size += encoder_dim

        self.lstm_cell = nn.LSTMCell(
            input_size=lstm_input_size,
            hidden_size=decoder_hidden_dim,
        )

        output_size = (
            decoder_hidden_dim
            + embedding_dim
        )

        if use_attention:
            output_size += encoder_dim

        self.output_projection = nn.Linear(
            output_size,
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

        context: Tensor | None = None

        if self.attention is not None:
            context, attention_weights = self.attention(
                hidden,
                encoder_outputs,
                source_mask,
            )

            lstm_input = torch.cat(
                (embedded, context),
                dim=1,
            )
        else:
            lstm_input = embedded

            attention_weights = encoder_outputs.new_zeros(
                (
                    encoder_outputs.size(0),
                    encoder_outputs.size(1),
                )
            )

        next_hidden, next_cell = self.lstm_cell(
            lstm_input,
            (hidden, cell),
        )

        if context is not None:
            output_features = torch.cat(
                (next_hidden, context, embedded),
                dim=1,
            )
        else:
            output_features = torch.cat(
                (next_hidden, embedded),
                dim=1,
            )

        logits = self.output_projection(
            output_features
        )

        return (
            logits,
            next_hidden,
            next_cell,
            attention_weights,
        )