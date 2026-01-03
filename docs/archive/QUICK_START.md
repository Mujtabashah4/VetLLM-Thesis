# VetLLM Quick Start Guide

**Quick reference for using the VetLLM pipeline**

---

## Prerequisites

```bash
# Install dependencies
pip install torch transformers datasets peft bitsandbytes accelerate
```

---

## Step 1: Validate Data

```bash
python scripts/validate_data.py
```

This validates all data files in `processed_data/` directory.

---

## Step 2: Train Model

### Basic Training

```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5
```

### With Validation Data

```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --val-data-path data/processed/val_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 3
```

### Disable 8-bit Quantization (if issues)

```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --output-dir models/vetllm-finetuned \
    --no-8bit \
    --epochs 3
```

---

## Step 3: Run Inference

### Single Clinical Note

```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --note "Cow. Clinical presentation includes epistaxis and high fever."
```

### Batch Processing

Create `test_notes.json`:
```json
[
  {"note": "Cow. Clinical presentation includes epistaxis."},
  {"note": "Buffalo. Clinical presentation includes persistent diarrhea."}
]
```

Run:
```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --input-file test_notes.json \
    --output-file predictions.json \
    --extract-codes
```

---

## Complete Pipeline (All Steps)

```bash
python scripts/run_pipeline.py \
    --data-path processed_data/all_processed_data.json \
    --model-name wxjiao/alpaca-7b \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --clinical-note "Cow. Clinical presentation includes epistaxis."
```

---

## Common Commands

### Validate Only
```bash
python scripts/run_pipeline.py --validate-only --data-path processed_data/all_processed_data.json
```

### Train Only
```bash
python scripts/run_pipeline.py --train-only --data-path processed_data/all_processed_data.json --epochs 3
```

### Inference Only
```bash
python scripts/run_pipeline.py \
    --inference-only \
    --model-path models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --clinical-note "Cow. Clinical presentation includes epistaxis."
```

---

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 2`
- Disable 8-bit: `--no-8bit`

### Model Not Found
- Ensure model path is correct
- For LoRA models, specify `--base-model-name wxjiao/alpaca-7b`

### Poor Results
- Train for more epochs: `--epochs 5`
- Use more training data
- Check data quality with validation script

---

## Data Files

- **Training:** `processed_data/all_processed_data.json` (1,602 samples)
- **Cow/Buffalo:** `processed_data/Verified_DLO_data_-_(Cow_Buffalo)_processed.json` (746 samples)
- **Sheep/Goat:** `processed_data/Verified_DLO_data_(Sheep_Goat)_processed.json` (856 samples)

All files are validated and ready to use!

---

For detailed documentation, see `PIPELINE_IMPLEMENTATION_REPORT.md`

