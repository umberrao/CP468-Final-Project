# CP468 Final Project: LSTM vs. LLM Text Simplification

## Project Status

**In development.** The dataset pipeline, custom LSTM Seq2Seq model, attention mechanism, training system, checkpointing, and inference pipeline are complete and working.

The next major step is full GPU training, followed by evaluation, ablation experiments, LLM baselines, error analysis, and the final report/demo.

## Project Goal

This project compares a custom LSTM sequence-to-sequence model with attention against a modern Large Language Model (LLM) for English text simplification.

- **Input:** Complex English sentence
- **Output:** Simplified English sentence
- **Training data:** WikiAuto
- **Test data:** ASSET
- **Primary metric:** SARI
- **Secondary metric:** BLEU

All models will be evaluated using the same frozen ASSET test set.

## Model Architecture

The Seq2Seq model is implemented directly in PyTorch.

1. Token embeddings
2. Bidirectional LSTM encoder
3. Masked additive (Bahdanau-style) attention
4. LSTM decoder
5. Vocabulary output projection
6. Greedy autoregressive inference

Padding masks prevent the attention mechanism from attending to padded tokens.

## Dataset

Dataset source:

`GEM/wiki_auto_asset_turk`

### Final Dataset

| Split        | Examples |
| ------------ | -------: |
| Training     |   50,000 |
| Validation   |    2,000 |
| Test (ASSET) |      359 |

Data preparation results:

- Removed overlaps: **0**
- Leakage check: **PASSED**
- Vocabulary is constructed only from training data.
- ASSET provides multiple human simplification references.
- Dataset files are intentionally excluded from Git and reproduced using the preparation script.

Prepare the final dataset with:

```powershell
python scripts/prepare_data.py --train-size 50000 --validation-size 2000 --shuffle-buffer 10000 --output-dir data/raw/final
```

## Reproducibility

- Fixed random seed: **468**
- Pinned dependencies
- Deterministic dataset preparation
- Training-only vocabulary
- Padding and attention masks
- Gradient clipping
- Configuration-based experiments
- Validation tracking
- Best-model checkpointing
- Training-time recording

## Current Progress

- [x] Python/PyTorch environment
- [x] Pinned dependencies
- [x] Dataset download/preparation pipeline
- [x] Final 50,000 / 2,000 / 359 dataset
- [x] Data-leakage protection
- [x] Tokenization/detokenization
- [x] Vocabulary construction
- [x] Padding and batching
- [x] Bidirectional LSTM encoder
- [x] Additive attention
- [x] LSTM decoder
- [x] Complete Seq2Seq model
- [x] Greedy decoding
- [x] Cross-entropy loss
- [x] Gradient clipping
- [x] Training pipeline
- [x] Validation pipeline
- [x] Model checkpointing
- [x] Command-line inference
- [x] **22 automated tests passing**
- [x] End-to-end CPU smoke training
- [ ] Full GPU training
- [ ] No-attention ablation
- [ ] SARI evaluation
- [ ] BLEU evaluation
- [ ] LLM zero-shot baseline
- [ ] LLM few-shot baseline
- [ ] Two or more LLM prompt variants
- [ ] LLM runtime/cost analysis
- [ ] Side-by-side qualitative examples
- [ ] Error analysis
- [ ] Results/figures
- [ ] Final report
- [ ] Demo video

## Smoke-Test Results

A small experiment verified that the entire training and inference pipeline works before committing resources to full training.

| Item                        |      Result |
| --------------------------- | ----------: |
| Training examples           |         100 |
| Validation examples         |          20 |
| Test examples               |         359 |
| Epochs                      |           2 |
| Vocabulary size             |       1,441 |
| Trainable parameters        |     927,905 |
| PyTorch                     |  2.13.0+cpu |
| Hardware                    |         CPU |
| Training time               | 6.1 seconds |
| Epoch 1 train loss          |      7.2637 |
| Epoch 1 validation loss     |      7.1863 |
| Epoch 2 train loss          |      6.9717 |
| Epoch 2 validation loss     |      6.9942 |
| Final validation perplexity |     1090.27 |

The smoke model is intentionally undertrained and is not used as a final result. Its purpose was to verify dataset loading, training, validation, checkpointing, and inference.

## Final Model Configuration

The planned full experiment is defined in `configs/final.yaml`.

- Embedding dimension: 256
- Encoder hidden dimension: 256
- Decoder hidden dimension: 256
- Attention dimension: 256
- Dropout: 0.3
- Maximum vocabulary: 20,000
- Batch size: 32
- Epochs: 10
- Learning rate: 0.001
- Gradient clipping: 1.0
- Seed: 468

Full training will be performed using a GPU.

## Running the Project

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run all tests:

```powershell
python -m pytest -q
```

Current expected result:

```text
22 passed
```

Run the smoke training experiment:

```powershell
python -m scripts.train --config configs/smoke.yaml
```

Test inference:

```powershell
python -m scripts.simplify --text "The scientist conducted an extensive investigation into the complicated problem."
```

The smoke model is not expected to generate high-quality simplifications.

## Project Structure

```text
CP468-Final-Project/
├── checkpoints/
├── configs/
│   ├── final.yaml
│   └── smoke.yaml
├── data/
├── report/
│   └── figures/
├── results/
├── scripts/
│   ├── prepare_data.py
│   ├── simplify.py
│   └── train.py
├── src/
│   ├── models/
│   │   ├── attention.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   └── seq2seq.py
│   ├── data.py
│   ├── inference.py
│   ├── text.py
│   └── training.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Next Steps

1. Train the full attention model on GPU.
2. Build and train the no-attention ablation.
3. Evaluate both models using SARI and BLEU.
4. Run zero-shot and few-shot LLM baselines.
5. Compare quantitative and qualitative results.
6. Perform categorized error analysis.
7. Complete the report and demonstration video.
