from __future__ import annotations

import argparse
import json
import platform
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import (
    SimplificationDataset,
    build_vocabulary,
    collate_batch,
)
from src.models.seq2seq import Seq2Seq
from src.training import (
    count_trainable_parameters,
    run_epoch,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = (
        config["training"].get(
            "mixed_precision",
            False,
        )
        and device.type == "cuda"
    )

    train_path = Path(config["data"]["train_path"])
    validation_path = Path(
        config["data"]["validation_path"]
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run prepare_data.py first."
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"{validation_path} not found."
        )

    output_directory = Path(
        config["output"]["directory"]
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary = build_vocabulary(
        train_path,
        min_frequency=config["data"][
            "vocab_min_frequency"
        ],
        max_size=config["data"]["vocab_max_size"],
    )

    vocabulary.save(
        output_directory / "vocabulary.json"
    )

    train_dataset = SimplificationDataset(
        train_path,
        vocabulary,
        max_source_length=config["data"][
            "max_source_length"
        ],
        max_target_length=config["data"][
            "max_target_length"
        ],
    )

    validation_dataset = SimplificationDataset(
        validation_path,
        vocabulary,
        max_source_length=config["data"][
            "max_source_length"
        ],
        max_target_length=config["data"][
            "max_target_length"
        ],
    )

    collate = partial(
        collate_batch,
        pad_id=vocabulary.pad_id,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate,
    )

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    parameter_count = count_trainable_parameters(model)

    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(0)
    else:
        hardware = platform.processor() or "CPU"

    print(f"Device: {device}")
    print(f"Hardware: {hardware}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mixed precision: {use_amp}")
    print(f"Vocabulary size: {len(vocabulary):,}")
    print(f"Training examples: {len(train_dataset):,}")
    print(
        f"Validation examples: "
        f"{len(validation_dataset):,}"
    )
    print(f"Trainable parameters: {parameter_count:,}")

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()

    epochs = config["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        train_batches = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} train",
            leave=False,
        )

        train_result = run_epoch(
            model,
            train_batches,
            pad_id=vocabulary.pad_id,
            device=device,
            optimizer=optimizer,
            max_gradient_norm=config["training"][
                "max_gradient_norm"
            ],
            use_amp=use_amp,
        )

        validation_batches = tqdm(
            validation_loader,
            desc=f"Epoch {epoch}/{epochs} validation",
            leave=False,
        )

        validation_result = run_epoch(
            model,
            validation_batches,
            pad_id=vocabulary.pad_id,
            device=device,
            use_amp=use_amp,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_result.loss:.4f} | "
            f"val loss={validation_result.loss:.4f} | "
            f"val ppl={validation_result.perplexity:.2f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "validation_loss": validation_result.loss,
                "validation_perplexity":
                    validation_result.perplexity,
            }
        )

        if validation_result.loss < best_validation_loss:
            best_validation_loss = validation_result.loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_loss":
                        validation_result.loss,
                    "config": config,
                },
                output_directory / "best.pt",
            )

    elapsed = time.perf_counter() - start_time

    (output_directory / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    summary = {
        "seed": seed,
        "device": str(device),
        "hardware": hardware,
        "torch_version": torch.__version__,
        "mixed_precision": use_amp,
        "parameter_count": parameter_count,
        "training_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }

    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Training time: {elapsed:.1f} seconds")
    print(
        f"Checkpoint: "
        f"{output_directory / 'best.pt'}"
    )


if __name__ == "__main__":
    main()