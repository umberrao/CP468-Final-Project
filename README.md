# CP468 Final Project: LSTM vs. LLM Text Simplification

## Project Status

**In development.** The dataset and preprocessing pipeline are complete. Model implementation, training, evaluation, and reporting are still in progress.

## Project Goal

This project compares a custom LSTM sequence-to-sequence model with attention against a modern large language model on text simplification.

- Input: complicated English sentence
- Output: simplified sentence
- Training data: WikiAuto
- Test data: ASSET
- Primary metric: SARI
- Secondary metric: BLEU

## Current Progress

- [x] Python and PyTorch environment
- [x] Pinned dependencies
- [x] Dataset download and preparation script
- [x] Fixed training, validation, and test splits
- [x] Data-leakage checks
- [x] Tokenization and detokenization
- [x] Training-only vocabulary construction
- [x] Padding and masking
- [x] Automated tests: 7 passing
- [ ] Bidirectional LSTM encoder
- [ ] Attention mechanism
- [ ] LSTM decoder
- [ ] Training and inference pipeline
- [ ] No-attention ablation
- [ ] SARI and BLEU evaluation
- [ ] LLM zero-shot and few-shot baselines
- [ ] Error analysis and result figures
- [ ] Final report and demonstration video

## Current Smoke-Test Dataset

- Training: 100 examples
- Validation: 20 examples
- Test: 359 examples
- Leakage detected: 0

The final experiment will use a larger training and validation sample.

## Run the Tests

```powershell
python -m pytest -q
```
