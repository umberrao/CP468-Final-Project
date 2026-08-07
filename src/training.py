from __future__ import annotations

import math
import os
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


@dataclass(frozen=True)
class EpochResult:
    loss: float
    perplexity: float
    token_count: int


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def sequence_cross_entropy(
    logits: Tensor,
    target_ids: Tensor,
    pad_id: int,
) -> Tensor:
    if logits.shape[:2] != target_ids.shape:
        raise ValueError(
            "Logit batch/sequence dimensions must match targets."
        )

    token_count = target_ids.ne(pad_id).sum()

    if token_count.item() == 0:
        raise ValueError("Batch contains no target tokens.")

    total_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )

    return total_loss / token_count


def run_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    pad_id: int,
    device: torch.device,
    optimizer: Optimizer | None = None,
    max_gradient_norm: float = 1.0,
    use_amp: bool = False,
) -> EpochResult:
    training = optimizer is not None
    amp_enabled = use_amp and device.type == "cuda"

    model.train(training)

    total_loss = 0.0
    total_tokens = 0

    scaler = (
        torch.amp.GradScaler("cuda")
        if training and amp_enabled
        else None
    )

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in batches:
            source_ids = batch["source_ids"].to(device)
            source_lengths = batch["source_lengths"].to(device)
            source_mask = batch["source_mask"].to(device)
            target_input_ids = batch["target_input_ids"].to(device)
            target_output_ids = batch["target_output_ids"].to(
                device
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, _ = model(
                    source_ids,
                    source_lengths,
                    source_mask,
                    target_input_ids,
                )

                loss = sequence_cross_entropy(
                    logits,
                    target_output_ids,
                    pad_id,
                )

            token_count = int(
                target_output_ids.ne(pad_id).sum().item()
            )

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    clip_grad_norm_(
                        model.parameters(),
                        max_gradient_norm,
                    )

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    loss.backward()

                    clip_grad_norm_(
                        model.parameters(),
                        max_gradient_norm,
                    )

                    optimizer.step()

            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("Epoch contained no target tokens.")

    average_loss = total_loss / total_tokens
    perplexity = (
        math.exp(average_loss)
        if average_loss < 709
        else float("inf")
    )

    return EpochResult(
        loss=average_loss,
        perplexity=perplexity,
        token_count=total_tokens,
    )