# VetLLM Experiments

Fine-tuning Large Language Models for Veterinary Clinical Diagnosis

---

## 🎯 Overview

This directory contains the complete experimental framework for fine-tuning LLMs on veterinary clinical data for disease diagnosis prediction in livestock (Cow, Buffalo, Sheep, Goat).

### Models

| Model | Role | Status |
|-------|------|--------|
| **Llama 3.1 8B Instruct** | Primary baseline | ✅ Ready |
| **Qwen2.5 7B Instruct** | Secondary comparison | ✅ Ready |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Raw clinical records | 1,602 |
| **Unique cases (after deduplication)** | **533** |
| Training samples | 373 (70%) |
| Validation samples | 80 (15%) |
| Test samples | 80 (15%) |
| Diseases covered | 22 unique conditions |
| Species | Cow, Buffalo, Sheep, Goat |

> **Note**: 1,069 duplicate entries (67%) were removed to prevent data leakage between train/test splits. See [EXPERIMENTAL_PLAN.md](EXPERIMENTAL_PLAN.md) for details.

---

## 📁 Directory Structure

```
experiments/
├── README.md                     # This file
├── EXPERIMENTAL_PLAN.md          # Detailed experimental methodology
├── requirements.txt              # Python dependencies
│
├── shared/                       # Shared utilities
│   ├── data_preprocessor.py      # Data preprocessing pipeline
│   ├── train.py                  # Unified training script
│   ├── inference.py              # Inference utilities
│   ├── evaluation/
│   │   └── evaluate.py           # Evaluation pipeline
│   └── utils/
│       └── __init__.py
│
├── llama3.1-8b/                  # Llama 3.1 experiment (PRIMARY)
│   ├── configs/
│   │   └── training_config.yaml  # Training configuration
│   ├── data/                     # Preprocessed data ✅ READY
│   │   ├── train.json            # 373 unique samples
│   │   ├── validation.json       # 80 unique samples
│   │   ├── test.json             # 80 unique samples
│   │   └── dataset_stats.json    # Disease/species distribution
│   ├── checkpoints/              # Model checkpoints (after training)
│   ├── results/                  # Evaluation results
│   ├── logs/                     # Training logs
│   └── run_experiment.sh         # One-click runner
│
└── qwen2.5-7b/                   # Qwen2.5 experiment (SECONDARY)
    ├── configs/
    │   └── training_config.yaml
    ├── data/                     # Preprocessed data ✅ READY
    │   ├── train.json            # 373 unique samples
    │   ├── validation.json       # 80 unique samples
    │   ├── test.json             # 80 unique samples
    │   └── dataset_stats.json
    ├── checkpoints/
    ├── results/
    ├── logs/
    └── run_experiment.sh
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r experiments/requirements.txt
```

### 2. Run Llama 3.1 8B Experiment

```bash
cd experiments/llama3.1-8b
./run_experiment.sh all
```

This will:
1. ✅ Preprocess data (already done)
2. Train the model with QLoRA
3. Evaluate on test set
4. Generate comparison report

### 3. Run Qwen2.5 7B Experiment (after Llama)

```bash
cd experiments/qwen2.5-7b
./run_experiment.sh all
```

---

## 📊 Data Format

### Llama 3.1 Format

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are VetLLM, a veterinary clinical assistant...
<|eot_id|><|start_header_id|>user<|end_header_id|>

Species: Cow
Clinical presentation: fever and diarrhea
...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1. **Primary Diagnosis**: **Disease Name** (SNOMED-CT: code)
...
<|eot_id|>
```

### Qwen2.5 Format (ChatML)

```
<|im_start|>system
You are VetLLM, a veterinary clinical assistant...
<|im_end|>
<|im_start|>user
**Species**: Cow
**Clinical Signs**: fever and diarrhea
...
<|im_end|>
<|im_start|>assistant
1. **Primary Diagnosis**: **Disease Name** (SNOMED-CT: code)
...
<|im_end|>
```

---

## 📈 Evaluation Metrics

### Classification
- Accuracy (exact match on diagnosis)
- F1 Score (macro/micro/weighted)
- Precision & Recall

### Generation
- BLEU Score
- ROUGE-1/2/L

---

## 🔧 Manual Commands

### Preprocess Data

```bash
# For Llama 3.1
python experiments/shared/data_preprocessor.py \
    --dataset-dir Dataset_UVAS \
    --output-dir experiments/llama3.1-8b/data \
    --model llama3.1

# For Qwen2.5
python experiments/shared/data_preprocessor.py \
    --dataset-dir Dataset_UVAS \
    --output-dir experiments/qwen2.5-7b/data \
    --model qwen2.5
```

### Train Model

```bash
python experiments/shared/train.py \
    --config experiments/llama3.1-8b/configs/training_config.yaml
```

### Evaluate Model

```bash
python experiments/shared/evaluation/evaluate.py \
    --model-path experiments/llama3.1-8b/checkpoints/final \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --test-data experiments/llama3.1-8b/data/test.json \
    --output-dir experiments/llama3.1-8b/results
```

### Interactive Inference

```bash
python experiments/shared/inference.py interactive \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --adapter experiments/llama3.1-8b/checkpoints/final \
    --model-type llama3.1
```

---

## 💻 Hardware Requirements

| Setup | Llama 3.1 8B | Qwen2.5 7B |
|-------|--------------|------------|
| **Min GPU VRAM** | 16 GB | 14 GB |
| **Recommended** | 24 GB | 24 GB |
| **RAM** | 32 GB | 32 GB |

### Supported GPUs
- NVIDIA A100 (40GB/80GB)
- NVIDIA RTX 4090 (24GB)
- NVIDIA RTX 3090 (24GB)
- NVIDIA V100 (32GB)

---

## 📝 Notes

1. **HuggingFace Login Required**: Run `huggingface-cli login` before downloading models

2. **Wandb (Optional)**: Set `report_to: "none"` in config to disable

3. **Memory Issues**: Reduce batch size or enable gradient checkpointing

---

## 📚 Documentation

- [Detailed Experimental Plan](EXPERIMENTAL_PLAN.md)
- [Project Documentation](../docs/README.md)

---

*Last Updated: January 2026*

