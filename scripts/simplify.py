from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.inference import load_model, simplify_text
from src.text import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/smoke/best.pt"),
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=Path(
            "checkpoints/smoke/vocabulary.json"
        ),
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=80,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocabulary = Vocabulary.load(args.vocabulary)

    model, config = load_model(
        args.checkpoint,
        vocabulary,
        device,
    )

    simplified, _ = simplify_text(
        model,
        vocabulary,
        args.text,
        device,
        max_source_length=config["data"][
            "max_source_length"
        ],
        max_output_length=args.max_output_length,
    )

    print(f"Input:  {args.text}")
    print(f"Output: {simplified}")


if __name__ == "__main__":
    main()