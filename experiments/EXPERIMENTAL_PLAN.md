# VetLLM Experimental Plan

## Dissertation Research: Fine-tuning LLMs for Veterinary Clinical Diagnosis

---

## Executive Summary

This document outlines the experimental plan for fine-tuning Large Language Models (LLMs) on veterinary clinical data for disease diagnosis prediction. We adopt a **two-model comparison strategy** using modern, strong baseline models rather than legacy architectures.

### Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Primary Model** | Llama 3.1 8B Instruct | Strong baseline, excellent documentation, widely recognized in benchmarks |
| **Secondary Model** | Qwen2.5 7B Instruct | State-of-the-art reasoning, excellent on medical tasks |
| **Deprecated** | Alpaca-7B | Legacy model, far weaker than modern 7-8B models |
| **Fine-tuning Method** | QLoRA (4-bit) | Memory-efficient, enables training on consumer GPUs |

---

## Phase 0: Dataset & Task Definition

### Task Definition

**Primary Task**: Veterinary Diagnosis Prediction
- Input: Clinical case presentation (species, symptoms, clinical signs)
- Output: 
  1. Primary diagnosis with SNOMED-CT code
  2. Differential diagnoses
  3. Treatment recommendations
  4. Clinical reasoning

### Data Schema

```json
{
  "animal": "Cow|Buffalo|Goat|Sheep",
  "symptoms": ["symptom1", "symptom2", ...],
  "disease": "Disease Name",
  "disease_normalized": "Standardized Disease Name",
  "snomed_codes": ["SNOMED-CT codes"],
  "source_file": "Original Excel file",
  "source_row": 123
}
```

### Data Sources

| File | Species | Description |
|------|---------|-------------|
| `Verified DLO data - (Cow Buffalo).xlsx` | Cow, Buffalo | Large ruminant diseases (778 rows) |
| `Verified DLO data (Sheep Goat).xlsx` | Sheep, Goat | Small ruminant diseases (859 rows) |

### Data Preprocessing & Deduplication

The raw dataset contained **1,602 clinical records**. Analysis revealed significant duplication:

| Statistic | Value |
|-----------|-------|
| Raw entries | 1,602 |
| Unique cases | 533 |
| Duplicate entries | 1,069 (67%) |

**Deduplication Rationale:**

Many entries were exact duplicates (identical animal + disease + symptom combinations). For example:
- "Sheep + PPR + fever" appeared 38 times
- "Goat + Kataa + 1 symptom" appeared 37 times

These are data entry duplicates, not distinct clinical cases. Keeping them would cause:

1. **Data Leakage**: Same case appearing in both training and test sets
2. **Overfitting**: Model memorizing specific cases instead of learning patterns
3. **Inflated Metrics**: Artificially high accuracy from memorization

**Decision**: Remove duplicates **before** splitting to ensure honest evaluation.

### Final Dataset Statistics

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 373 | 70% |
| Validation | 80 | 15% |
| Test | 80 | 15% |
| **Total** | **533** | 100% |

### Disease Distribution (Training Set)

| Disease | Count | % |
|---------|-------|---|
| Peste des Petits Ruminants (PPR) | 122 | 32.7% |
| Foot and Mouth Disease (FMD) | 56 | 15.0% |
| Mastitis | 48 | 12.9% |
| Hemorrhagic Septicemia | 42 | 11.3% |
| Black Quarter | 29 | 7.8% |
| CCPP | 29 | 7.8% |
| Anthrax | 15 | 4.0% |
| Other diseases (15 types) | 32 | 8.5% |

### Species Distribution (Training Set)

| Species | Count | % |
|---------|-------|---|
| Goat | 103 | 27.6% |
| Sheep | 100 | 26.8% |
| Cow | 90 | 24.1% |
| Buffalo | 80 | 21.5% |

### Data Splits

- **Training**: 70% (373 samples)
- **Validation**: 15% (80 samples)
- **Test**: 15% (80 samples)
- **Splitting Strategy**: Stratified by disease class, case-level splitting after deduplication to prevent data leakage

---

## Thesis Methodology Section (Copy-Ready)

The following text can be adapted for your dissertation's methodology chapter:

> ### 3.X Data Collection and Preprocessing
>
> The dataset was obtained from the University of Veterinary and Animal Sciences (UVAS) Punjab, Pakistan, comprising clinical records from the Digital Livestock Officer (DLO) system. The raw dataset contained **1,602 clinical entries** across two Excel files:
>
> - **Large Ruminants**: 778 records (Cow, Buffalo)
> - **Small Ruminants**: 859 records (Sheep, Goat)
>
> Each record contained: animal species, disease diagnosis, and binary indicators for 19-27 clinical symptoms.
>
> **Data Cleaning**: Analysis revealed that 67% of entries (1,069 records) were exact duplicates—identical combinations of animal species, disease, and symptom patterns. These represent data entry redundancies rather than distinct clinical cases. To prevent data leakage between training and evaluation sets, we removed duplicate entries, yielding **533 unique clinical cases** across 22 disease categories.
>
> **Data Splitting**: The deduplicated dataset was split using stratified sampling by disease class:
> - Training set: 373 cases (70%)
> - Validation set: 80 cases (15%)  
> - Test set: 80 cases (15%)
>
> This preprocessing ensures that no identical case appears in both training and test sets, providing an honest assessment of model generalization capability.

---

## Phase 1: Llama 3.1 8B Baseline

### Model Configuration

```yaml
Model: meta-llama/Llama-3.1-8B-Instruct
Quantization: 4-bit (NF4)
Fine-tuning: QLoRA
Context Window: 512 tokens (extendable to 128K)
```

### Prompt Format (Llama 3.1)

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are VetLLM, a veterinary clinical assistant specialized in livestock diseases...
<|eot_id|><|start_header_id|>user<|end_header_id|>

Species: {species}
Clinical presentation: {symptoms}
...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1. **Primary Diagnosis**: {diagnosis}
2. **Differential Diagnoses**: ...
3. **Recommended Treatment**: ...
4. **Clinical Reasoning**: ...
<|eot_id|>
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Batch Size | 4 (effective: 16 with grad accum) |
| Epochs | 3 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Optimizer | Paged AdamW 8-bit |
| Scheduler | Cosine |

### Expected Outputs

1. Fine-tuned model adapter (`checkpoints/final/`)
2. Training metrics (`training_metrics.json`)
3. Validation loss curves

---

## Phase 2: Qwen2.5 7B Comparison

### Model Configuration

```yaml
Model: Qwen/Qwen2.5-7B-Instruct
# Alternatives:
# - HPAI-BSC/Qwen2.5-Aloe-Beta-7B (Medical variant)
# - Lightblue/Meditron3-Qwen2.5-7B (Medical variant)
Quantization: 4-bit (NF4)
Fine-tuning: QLoRA
```

### Prompt Format (Qwen2.5 ChatML)

```
<|im_start|>system
You are VetLLM, a veterinary clinical assistant...
<|im_end|>
<|im_start|>user
Analyze this veterinary case:
**Species**: {species}
**Clinical Signs**: {symptoms}
...
<|im_end|>
<|im_start|>assistant
1. **Primary Diagnosis**: ...
<|im_end|>
```

### Training Configuration

Identical to Llama 3.1 to ensure fair comparison.

---

## Phase 3: Evaluation Pipeline

### Metrics

#### Classification Metrics
- **Accuracy**: Exact match on primary diagnosis
- **F1 Score** (macro/micro/weighted)
- **Precision & Recall**

#### Generation Metrics
- **BLEU**: N-gram overlap with reference
- **ROUGE-1/2/L**: Unigram/bigram/longest common subsequence overlap
- **BERTScore**: Semantic similarity (optional)

#### Clinical Metrics (Manual Evaluation)
- **Clinical Plausibility**: Is the diagnosis reasonable?
- **Safety**: No dangerous recommendations?
- **Completeness**: All relevant differentials covered?

### Comparison Matrix

| Model | Condition | Accuracy | F1 | BLEU | ROUGE-L |
|-------|-----------|----------|-----|------|---------|
| Llama 3.1 8B | Zero-shot | - | - | - | - |
| Llama 3.1 8B | Fine-tuned | - | - | - | - |
| Qwen2.5 7B | Zero-shot | - | - | - | - |
| Qwen2.5 7B | Fine-tuned | - | - | - | - |

---

## Directory Structure

```
experiments/
├── EXPERIMENTAL_PLAN.md          # This document
├── shared/
│   ├── data_preprocessor.py      # Unified data preprocessing
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference utilities
│   ├── evaluation/
│   │   └── evaluate.py           # Evaluation pipeline
│   └── utils/
├── llama3.1-8b/
│   ├── configs/
│   │   └── training_config.yaml  # Model-specific config
│   ├── data/                     # Preprocessed data (Llama format)
│   ├── checkpoints/              # Model checkpoints
│   ├── results/                  # Evaluation results
│   ├── logs/                     # Training logs
│   └── run_experiment.sh         # One-click experiment runner
└── qwen2.5-7b/
    ├── configs/
    │   └── training_config.yaml
    ├── data/                     # Preprocessed data (Qwen format)
    ├── checkpoints/
    ├── results/
    ├── logs/
    └── run_experiment.sh
```

---

## Quick Start Guide

### Prerequisites

```bash
# Install dependencies
pip install torch>=2.1.0 transformers>=4.35.0 peft>=0.6.0 \
            bitsandbytes>=0.41.0 accelerate>=0.24.0 \
            datasets scikit-learn pandas wandb

# For generation metrics
pip install nltk rouge-score
```

### Running Experiments

#### Option 1: Full Pipeline (Recommended)

```bash
# Llama 3.1 8B experiment
cd experiments/llama3.1-8b
chmod +x run_experiment.sh
./run_experiment.sh all

# Qwen2.5 7B experiment (after Llama completes)
cd ../qwen2.5-7b
chmod +x run_experiment.sh
./run_experiment.sh all
```

#### Option 2: Step-by-Step

```bash
# Step 1: Preprocess data
./run_experiment.sh preprocess

# Step 2: Train model
./run_experiment.sh train

# Step 3: Evaluate
./run_experiment.sh evaluate

# Step 4: Generate report
./run_experiment.sh report
```

#### Option 3: Manual Execution

```bash
# Preprocess for Llama 3.1
python experiments/shared/data_preprocessor.py \
    --dataset-dir Dataset_UVAS \
    --output-dir experiments/llama3.1-8b/data \
    --model llama3.1

# Train
python experiments/shared/train.py \
    --config experiments/llama3.1-8b/configs/training_config.yaml

# Evaluate
python experiments/shared/evaluation/evaluate.py \
    --model-path experiments/llama3.1-8b/checkpoints/final \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --test-data experiments/llama3.1-8b/data/test.json \
    --output-dir experiments/llama3.1-8b/results
```

---

## Compute Requirements

### Minimum Requirements

| Resource | Llama 3.1 8B (QLoRA) | Qwen2.5 7B (QLoRA) |
|----------|---------------------|-------------------|
| GPU VRAM | 16 GB | 14 GB |
| RAM | 32 GB | 32 GB |
| Storage | 50 GB | 50 GB |

### Recommended Setup

- **GPU**: NVIDIA A100 40GB, RTX 4090 24GB, or RTX 3090 24GB
- **RAM**: 64 GB
- **Storage**: 100 GB SSD

### Estimated Training Time

| Setup | Llama 3.1 8B | Qwen2.5 7B |
|-------|--------------|------------|
| A100 40GB | ~2 hours | ~1.5 hours |
| RTX 4090 | ~3 hours | ~2.5 hours |
| RTX 3090 | ~4 hours | ~3.5 hours |

---

## Dissertation Narrative

The final dissertation can follow this structure:

1. **Introduction**: Problem statement - veterinary diagnosis in resource-limited settings

2. **Related Work**: LLMs in medical/veterinary domains, fine-tuning approaches

3. **Methodology**:
   - "We selected Llama 3.1 8B as a strong open baseline, widely recognized in academic benchmarks"
   - "Qwen2.5 7B serves as a reasoning-focused comparison, known for strong performance on medical tasks"
   - "Both models were fine-tuned using QLoRA on identical veterinary data splits"

4. **Experiments**:
   - Zero-shot baseline evaluation
   - Fine-tuned model evaluation
   - Cross-model comparison

5. **Results**:
   - "Fine-tuning significantly improves diagnosis accuracy over zero-shot baselines"
   - "[Model X] achieves best performance on [metrics]"
   - Ablation studies (if time permits)

6. **Discussion & Conclusion**:
   - Practical implications for veterinary practice
   - Limitations and future work

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size` to 2
   - Increase `gradient_accumulation_steps` to 8
   - Enable gradient checkpointing (already enabled by default)

2. **Model Download Fails**
   - Ensure HuggingFace token is configured: `huggingface-cli login`
   - Accept model license on HuggingFace website

3. **Training Loss Not Decreasing**
   - Check data format matches expected chat template
   - Verify tokenizer special tokens are correct
   - Try reducing learning rate to 5e-5

4. **Evaluation Metrics Are Zero**
   - Check model output format matches expected structure
   - Verify test data has correct fields

---

## Contact & Support

For questions about this experimental setup, refer to:
- HuggingFace Transformers documentation
- PEFT library documentation
- Project README files

---

*Last Updated: January 2026*

