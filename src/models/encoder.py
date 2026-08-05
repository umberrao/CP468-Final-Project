from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        pad_id: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        combined_size = encoder_hidden_dim * 2

        self.hidden_projection = nn.Linear(
            combined_size,
            decoder_hidden_dim,
        )
        self.cell_projection = nn.Linear(
            combined_size,
            decoder_hidden_dim,
        )

    def forward(
        self,
        source_ids: Tensor,
        source_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embedded = self.embedding_dropout(
            self.embedding(source_ids)
        )

        packed = pack_padded_sequence(
            embedded,
            source_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_outputs, (hidden, cell) = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=source_ids.size(1),
        )

        final_hidden = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1,
        )
        final_cell = torch.cat(
            (cell[-2], cell[-1]),
            dim=1,
        )

        decoder_hidden = torch.tanh(
            self.hidden_projection(final_hidden)
        )
        decoder_cell = torch.tanh(
            self.cell_projection(final_cell)
        )

        return encoder_outputs, decoder_hidden, decoder_cell