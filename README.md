# CP468 Final Project: LSTM vs. LLM Text Simplification

## Project Status

The data pipeline, custom PyTorch models, GPU training, ablation,
evaluation system, and local LLM experiments are complete. All six
evaluated configurations used the same frozen 359-example ASSET test
set with 10 human references per example.

Remaining work consists of qualitative/error analysis, figures, the
five-page report, and the eight-minute demonstration video.

## Project Goal

This project compares a custom LSTM sequence-to-sequence model with
attention against a modern instruction-tuned Large Language Model for
English text simplification.

- **Input:** Complex English text
- **Output:** Simplified English text
- **Training data:** WikiAuto
- **Test data:** ASSET
- **Primary metric:** SARI
- **Secondary metric:** BLEU
- **Random seed:** 468

## Final Results

| System                 | Prompt setting        |        SARI |        BLEU | Recorded evaluation/generation time |
| ---------------------- | --------------------- | ----------: | ----------: | ----------------------------------: |
| LSTM without attention | N/A                   |     22.9017 |      1.9097 |                               5.4 s |
| LSTM with attention    | N/A                   |     35.5013 |     39.3457 |                               7.7 s |
| Qwen2.5-7B-Instruct    | Direct zero-shot      |     44.7089 |     69.7172 |                             308.6 s |
| Qwen2.5-7B-Instruct    | Controlled zero-shot  |     46.8721 |     62.3587 |                             384.3 s |
| Qwen2.5-7B-Instruct    | Direct three-shot     |     45.1292 |     76.6130 |                             396.7 s |
| Qwen2.5-7B-Instruct    | Controlled three-shot | **49.7276** | **81.7623** |                             496.8 s |

The attention mechanism improved SARI by 12.5996 points over the
no-attention ablation. The best LLM setting improved SARI by 14.2263
points over the attention model. SARI is treated as the main result;
BLEU is reported as a secondary lexical-overlap metric.

Machine-readable values are stored in
`results/final_metrics.json`.

## Dataset

The data is prepared from `GEM/wiki_auto_asset_turk` using streaming
downloads from Hugging Face.

| Split      | Source   | Examples | References per example |
| ---------- | -------- | -------: | ---------------------: |
| Training   | WikiAuto |   50,000 |                      1 |
| Validation | WikiAuto |    2,000 |                      1 |
| Test       | ASSET    |      359 |                     10 |

Data preparation produced zero source overlaps between training,
validation, and test data. The vocabulary is built only from the
training split. Generated dataset files are excluded from Git and can
be reproduced with the preparation script.

Dataset licenses recorded by the manifest:

- WikiAuto train/validation: CC BY-NC 3.0
- ASSET test: CC BY-NC 4.0

## Custom Model Architecture

The sequence-to-sequence models are implemented directly in PyTorch:

1. Token embeddings
2. Bidirectional LSTM encoder
3. Optional masked additive attention
4. LSTM decoder
5. Vocabulary output projection
6. Greedy autoregressive inference

Padding masks prevent attention from using padded tokens. The
no-attention model retains the same data, training, and decoding setup
while removing the attention component.

### Shared Configuration

- Embedding dimension: 256
- Encoder hidden dimension: 256
- Decoder hidden dimension: 256
- Attention dimension: 256
- Dropout: 0.3
- Maximum vocabulary: 20,000
- Maximum source/target length: 80
- Batch size: 32
- Maximum epochs: 10
- Learning rate: 0.001
- Gradient clipping: 1.0
- Mixed precision: enabled on CUDA
- Hardware: Tesla T4

### Training Outcomes

| Model        | Parameters | Best epoch | Best validation loss | Best validation perplexity |                    Training time |
| ------------ | ---------: | ---------: | -------------------: | -------------------------: | -------------------------------: |
| Attention    | 33,302,816 |          4 |               2.7854 |                      16.21 | Not captured for interrupted run |
| No attention | 22,341,664 |          5 |               4.7791 |                     118.99 |                        2,251.7 s |

The attention session ended after epoch 7, but best-checkpoint saving
preserved epoch 4. Validation loss had already stopped improving after
that epoch. The missing exact attention training time must be noted or
resolved before the final report is submitted.

## Local LLM Baseline

The comparison model is `Qwen/Qwen2.5-7B-Instruct`, run locally on a
Tesla T4 using 4-bit NF4 quantization. No paid API was used, so the API
cost was $0.

Four required prompt settings were evaluated:

- Direct zero-shot
- Controlled zero-shot
- Direct three-shot
- Controlled three-shot

The three demonstrations are fixed WikiAuto training examples and are
never taken from the test set. Exact system prompts, user templates,
and demonstrations are defined in `src/llm.py` and saved with each
evaluation result. Decoding is deterministic and greedy.

The four full evaluations required 1,586.3 seconds of generation time,
or approximately 0.441 Tesla T4 GPU-hours.

LLM environment:

- Transformers 5.0.0
- Accelerate 1.13.0
- BitsAndBytes 0.50.0
- PyTorch 2.10.0+cu128

## Reproducibility

- Fixed seed of 468
- Pinned local and LLM dependencies
- Deterministic dataset preparation
- Training-only vocabulary construction
- Leakage validation and dataset hashes
- Configuration-based model experiments
- Gradient clipping and mixed-precision support
- Validation tracking and best-model checkpointing
- Exact prompt definitions and decoding settings
- Resumable LLM evaluation
- Saved predictions, per-example SARI, aggregate metrics, and runtime
- **45 automated tests passing**

## Running the Project

Run commands from the repository root.

### Local Development Environment

The main requirements use CPU-only PyTorch for Windows development:

```powershell
python -m pip install -r requirements.txt
```

### Prepare the Final Dataset

```powershell
python scripts/prepare_data.py --train-size 50000 --validation-size 2000 --shuffle-buffer 10000 --output-dir data/raw/final
```

### Run Tests

```powershell
python -m pytest -q
```

Expected result:

```text
45 passed
```

### Train the Custom Models

These commands should be run in a CUDA-enabled environment:

```powershell
python -m scripts.train --config configs/final.yaml
python -m scripts.train --config configs/no_attention.yaml
```

### Evaluate the Attention Model

```powershell
python -m scripts.evaluate --test-data data/raw/final/test.jsonl --checkpoint checkpoints/final/best.pt --vocabulary checkpoints/final/vocabulary.json --output-dir results/attention
```

### Evaluate the No-Attention Model

```powershell
python -m scripts.evaluate --test-data data/raw/final/test.jsonl --checkpoint checkpoints/no_attention/best.pt --vocabulary checkpoints/no_attention/vocabulary.json --output-dir results/no_attention
```

### Run a Custom Simplification

```powershell
python -m scripts.simplify --text "The scientist conducted an extensive investigation into the complicated problem." --checkpoint checkpoints/final/best.pt --vocabulary checkpoints/final/vocabulary.json
```

### Run the Local Qwen Evaluation

Use a Colab GPU runtime. Colab supplies its CUDA-enabled PyTorch build;
do not replace it with the CPU-only PyTorch pin from `requirements.txt`.

```bash
python -m pip install -r requirements-llm.txt
python -m scripts.evaluate_llm --test-data data/raw/final/test.jsonl --output-dir results/qwen2_5_7b
```

The evaluator validates all 359 test records and all 10 references,
runs the four prompt settings, saves progress after every batch, and
resumes safely after an interruption.

## Project Structure

```text
CP468-Final-Project/
|-- checkpoints/
|-- configs/
|   |-- final.yaml
|   |-- no_attention.yaml
|   `-- smoke.yaml
|-- data/
|-- report/
|   `-- figures/
|-- results/
|   `-- final_metrics.json
|-- scripts/
|   |-- evaluate.py
|   |-- evaluate_llm.py
|   |-- prepare_data.py
|   |-- simplify.py
|   `-- train.py
|-- src/
|   |-- models/
|   |-- data.py
|   |-- inference.py
|   |-- llm.py
|   |-- metrics.py
|   |-- text.py
|   `-- training.py
|-- tests/
|-- requirements-llm.txt
|-- requirements.txt
`-- README.md
```

## Completed Work

- [x] Reproducible WikiAuto/ASSET data pipeline
- [x] Leakage-free 50,000/2,000/359 splits
- [x] Custom bidirectional LSTM Seq2Seq model
- [x] Masked additive attention
- [x] No-attention ablation
- [x] GPU training with mixed precision
- [x] SARI and BLEU implementation
- [x] Full 359-example evaluation with 10 references
- [x] Qwen zero-shot and three-shot baselines
- [x] Two exact prompt variants
- [x] LLM runtime and cost accounting
- [x] Reproducible LLM evaluation script
- [x] 45 automated tests

## Remaining Work

1. Resolve or document the missing exact attention training time.
2. Select at least 10 side-by-side qualitative examples.
3. Categorize and quantify model errors.
4. Generate final tables and figures.
5. Write the five-page report plus references/appendix.
6. Record and edit the eight-minute demonstration video.
