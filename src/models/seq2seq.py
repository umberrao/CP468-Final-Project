from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        pad_id: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        encoder_dim = encoder_hidden_dim * 2

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            pad_id=pad_id,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
            pad_id=pad_id,
            dropout=dropout,
        )

    def forward(
        self,
        source_ids: Tensor,
        source_lengths: Tensor,
        source_mask: Tensor,
        decoder_input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoder_outputs, hidden, cell = self.encoder(
            source_ids,
            source_lengths,
        )

        logits_steps = []
        attention_steps = []

        for step in range(decoder_input_ids.size(1)):
            logits, hidden, cell, attention = self.decoder(
                decoder_input_ids[:, step],
                hidden,
                cell,
                encoder_outputs,
                source_mask,
            )

            logits_steps.append(logits)
            attention_steps.append(attention)

        logits = torch.stack(logits_steps, dim=1)
        attention_weights = torch.stack(
            attention_steps,
            dim=1,
        )

        return logits, attention_weights

    @torch.no_grad()
    def greedy_decode(
        self,
        source_ids: Tensor,
        source_lengths: Tensor,
        source_mask: Tensor,
        bos_id: int,
        eos_id: int,
        max_length: int,
    ) -> tuple[Tensor, Tensor]:
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        encoder_outputs, hidden, cell = self.encoder(
            source_ids,
            source_lengths,
        )

        batch_size = source_ids.size(0)
        input_ids = torch.full(
            (batch_size,),
            bos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=source_ids.device,
        )

        generated_steps = []
        attention_steps = []

        for _ in range(max_length):
            logits, hidden, cell, attention = self.decoder(
                input_ids,
                hidden,
                cell,
                encoder_outputs,
                source_mask,
            )

            predicted_ids = logits.argmax(dim=1)
            predicted_ids = torch.where(
                finished,
                torch.full_like(predicted_ids, eos_id),
                predicted_ids,
            )

            generated_steps.append(predicted_ids)
            attention_steps.append(attention)

            finished = finished | predicted_ids.eq(eos_id)
            input_ids = predicted_ids

            if finished.all():
                break

        generated = torch.stack(generated_steps, dim=1)
        attention_weights = torch.stack(
            attention_steps,
            dim=1,
        )

        return generated, attention_weights