from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from tqdm.auto import tqdm

from src.data import iter_jsonl
from src.llm import (
    EXPERIMENT_SETTINGS,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    build_messages,
    parse_setting,
    prompt_record,
)
from src.metrics import (
    corpus_bleu,
    corpus_sari,
    sari_sentence,
)


EXPECTED_TEST_SIZE = 359
EXPECTED_REFERENCES = 10
SEED = 468


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen text-simplification prompts on ASSET."
        )
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/raw/final/test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen2_5_7b"),
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=EXPERIMENT_SETTINGS,
        default=list(EXPERIMENT_SETTINGS),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=MAX_INPUT_LENGTH,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
    )

    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_test_records(
    path: Path,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    records = list(iter_jsonl(path))

    if len(records) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SIZE} ASSET test "
            f"examples, found {len(records)}."
        )

    for index, record in enumerate(records):
        references = record.get("references", [])

        if len(references) != EXPECTED_REFERENCES:
            raise ValueError(
                f"Example {index} has {len(references)} "
                f"references; expected {EXPECTED_REFERENCES}."
            )

    if max_examples is not None:
        if max_examples < 1:
            raise ValueError(
                "--max-examples must be positive."
            )

        records = records[:max_examples]

    return records


def load_model_and_tokenizer(
    model_name: str,
) -> tuple[Any, Any, float]:
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the Colab dependencies with "
            "`python -m pip install -r "
            "requirements-llm.txt`."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for the 7B local LLM. "
            "In Colab, select Runtime > Change runtime type > T4 GPU."
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Greedy decoding does not use sampling parameters. Clearing them
    # prevents Transformers from printing irrelevant flag warnings.
    for attribute in (
        "temperature",
        "top_p",
        "top_k",
    ):
        if hasattr(model.generation_config, attribute):
            setattr(
                model.generation_config,
                attribute,
                None,
            )

    load_seconds = time.perf_counter() - start_time

    return model, tokenizer, load_seconds


def generate_batch(
    model: Any,
    tokenizer: Any,
    sources: list[str],
    variant: str,
    shots: int,
    max_input_length: int,
    max_new_tokens: int,
) -> tuple[list[str], float]:
    chats = [
        tokenizer.apply_chat_template(
            build_messages(source, variant, shots),
            tokenize=False,
            add_generation_prompt=True,
        )
        for source in sources
    ]

    encoded = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    prompt_length = encoded["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    predictions = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return [prediction.strip() for prediction in predictions], elapsed


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction_rows(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def verify_resume_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    if len(rows) > len(records):
        raise ValueError(
            "Saved predictions contain more rows than the test set."
        )

    for index, row in enumerate(rows):
        if row.get("source") != records[index]["source"]:
            raise ValueError(
                "Saved predictions do not match the current test "
                f"data at row {index}. Use a different output directory."
            )


def build_experiment_record(
    model_name: str,
    variant: str,
    shots: int,
    test_sha256: str,
    test_examples: int,
    max_input_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "prompt": prompt_record(variant, shots),
        "test_sha256": test_sha256,
        "test_examples": test_examples,
        "references_per_example": EXPECTED_REFERENCES,
        "decoding": {
            "method": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
        },
        "quantization": {
            "bits": 4,
            "type": "nf4",
            "double_quantization": True,
            "compute_dtype": "float16",
        },
        "seed": SEED,
    }


def run_setting(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    setting: str,
    output_root: Path,
    model_name: str,
    test_sha256: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    variant, shots = parse_setting(setting)
    setting_dir = output_root / setting
    setting_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = setting_dir / "prompt.json"
    predictions_path = setting_dir / "predictions.jsonl"
    progress_path = setting_dir / "progress.json"
    metrics_path = setting_dir / "metrics.json"

    experiment = build_experiment_record(
        model_name,
        variant,
        shots,
        test_sha256,
        len(records),
        max_input_length,
        max_new_tokens,
    )

    if prompt_path.exists():
        if read_json(prompt_path) != experiment:
            raise ValueError(
                f"{prompt_path} describes a different experiment. "
                "Use a new output directory or remove that setting's "
                "old result directory."
            )
    else:
        write_json(prompt_path, experiment)

    output_rows = read_prediction_rows(predictions_path)
    verify_resume_rows(output_rows, records)

    generation_seconds = 0.0

    if progress_path.exists():
        progress = read_json(progress_path)
        generation_seconds = float(
            progress.get("generation_seconds", 0.0)
        )

    with tqdm(
        total=len(records),
        initial=len(output_rows),
        desc=setting,
    ) as progress_bar:
        for start in range(
            len(output_rows),
            len(records),
            batch_size,
        ):
            batch = records[start:start + batch_size]
            predictions, elapsed = generate_batch(
                model,
                tokenizer,
                [record["source"] for record in batch],
                variant,
                shots,
                max_input_length,
                max_new_tokens,
            )
            generation_seconds += elapsed

            new_rows = []

            for offset, (record, prediction) in enumerate(
                zip(batch, predictions)
            ):
                references = record["references"]
                index = start + offset

                new_rows.append(
                    {
                        "id": record.get(
                            "id",
                            record.get("gem_id", str(index)),
                        ),
                        "source": record["source"],
                        "target": record.get("target", ""),
                        "prediction": prediction,
                        "references": references,
                        "sari": sari_sentence(
                            record["source"],
                            prediction,
                            references,
                        ),
                    }
                )

            with predictions_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                for row in new_rows:
                    file.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            output_rows.extend(new_rows)

            write_json(
                progress_path,
                {
                    "completed_examples": len(output_rows),
                    "total_examples": len(records),
                    "generation_seconds": generation_seconds,
                    "complete": (
                        len(output_rows) == len(records)
                    ),
                },
            )
            progress_bar.update(len(new_rows))

    sources = [record["source"] for record in records]
    references = [
        record["references"] for record in records
    ]
    predictions = [
        row["prediction"] for row in output_rows
    ]

    sari = corpus_sari(
        sources,
        predictions,
        references,
    )
    bleu = corpus_bleu(
        predictions,
        references,
    )

    metrics = {
        "model": model_name,
        "setting": setting,
        "prompt_variant": variant,
        "shots": shots,
        "test_examples": len(records),
        "full_test_set": (
            len(records) == EXPECTED_TEST_SIZE
        ),
        "references_per_example": EXPECTED_REFERENCES,
        "sari": sari,
        "bleu": bleu,
        "generation_seconds": generation_seconds,
        "api_cost_usd": 0.0,
        "execution": "local_open_weights",
        "device": "cuda",
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": package_version(
            "transformers"
        ),
        "accelerate_version": package_version("accelerate"),
        "bitsandbytes_version": package_version(
            "bitsandbytes"
        ),
        "sacrebleu_version": sacrebleu.__version__,
        "quantization": experiment["quantization"],
        "decoding": experiment["decoding"],
    }

    write_json(metrics_path, metrics)

    print()
    print(setting)
    print(f"Examples: {len(records)}")
    print(f"SARI:     {sari:.4f}")
    print(f"BLEU:     {bleu:.4f}")
    print(
        f"Time:     {generation_seconds:.1f} seconds"
    )
    print(f"Results:  {setting_dir}")

    return metrics


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    if args.max_new_tokens < 1:
        raise ValueError(
            "--max-new-tokens must be positive."
        )

    if args.max_input_length < 1:
        raise ValueError(
            "--max-input-length must be positive."
        )

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    records = load_test_records(
        args.test_data,
        args.max_examples,
    )
    test_sha256 = file_sha256(args.test_data)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, tokenizer, load_seconds = (
        load_model_and_tokenizer(args.model)
    )

    all_metrics = {}

    for setting in args.settings:
        all_metrics[setting] = run_setting(
            model,
            tokenizer,
            records,
            setting,
            args.output_dir,
            args.model,
            test_sha256,
            args.batch_size,
            args.max_input_length,
            args.max_new_tokens,
        )

    total_generation_seconds = sum(
        result["generation_seconds"]
        for result in all_metrics.values()
    )

    aggregate = {
        "model": args.model,
        "model_load_seconds": load_seconds,
        "total_generation_seconds": (
            total_generation_seconds
        ),
        "gpu_hours": total_generation_seconds / 3600.0,
        "api_cost_usd": 0.0,
        "settings": all_metrics,
    }

    write_json(
        args.output_dir / "all_metrics.json",
        aggregate,
    )

    print()
    print("All requested Qwen evaluations completed.")
    print(
        "Total generation: "
        f"{total_generation_seconds:.1f} seconds"
    )
    print(
        f"GPU-hours: {total_generation_seconds / 3600.0:.3f}"
    )


if __name__ == "__main__":
    main()
