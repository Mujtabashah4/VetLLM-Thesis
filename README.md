# VetLLM: Veterinary Diagnosis Prediction using LLMs

**Fine-tuning Large Language Models for SNOMED-CT Code Prediction from Veterinary Clinical Notes**

---

## 📋 Project Overview

VetLLM is a research project that fine-tunes large language models (LLMs) to predict SNOMED-CT diagnosis codes from veterinary clinical notes. The project has successfully trained and evaluated two models: **Alpaca-7B** and **QWEN 2.5-7B**, with plans to test **Llama3.1-8B**.

---

## 🎯 Key Results

### Models Tested
- ✅ **Alpaca-7B**: 40% accuracy, 46.15% F1 macro, better rare disease handling
- ✅ **QWEN 2.5-7B**: 50% accuracy, 16.44% F1 macro, excellent on common diseases
- ⏳ **Llama3.1-8B**: Planned for future testing

### Dataset
- **Training**: 373 samples
- **Validation**: 80 samples
- **Test**: 80 samples
- **Total**: 533 unique cases from UVAS DLO System

---

## 📁 Project Structure

```
VetLLM-Thesis/
├── models/              # Base models and fine-tuned checkpoints
│   ├── alpaca-7b/       # Alpaca base model
│   ├── alpaca-7b-native/ # Alpaca native model
│   ├── qwen2.5-7b-instruct/ # QWEN base model
│   └── vetllm-finetuned*/ # Fine-tuned models
│
├── experiments/          # Training experiments and results
│   ├── qwen2.5-7b/      # QWEN experiments
│   ├── llama3.1-8b/     # Llama3.1 experiments (future)
│   ├── checkpoints/      # Training checkpoints
│   └── shared/          # Shared training/evaluation code
│
├── reports/              # All reports and documentation
│   ├── alpaca/          # Alpaca-specific reports
│   ├── qwen/            # QWEN-specific reports
│   ├── llama3.1/        # Llama3.1 reports (future)
│   ├── comparison/      # Model comparison reports
│   ├── general/         # General project reports
│   └── REPORT_INDEX.md  # Complete documentation index
│
├── logs/                 # Training and evaluation logs
│   ├── alpaca/          # Alpaca logs
│   ├── qwen/            # QWEN logs
│   └── general/        # General logs
│
├── data/                 # Data files
│   └── snomed_codes.json # SNOMED code mappings
│
├── processed_data/       # Processed datasets
│   ├── all_processed_data.json
│   └── Verified_DLO_data_*.json
│
├── scripts/              # Utility scripts (organized by purpose)
│   ├── training/        # Training scripts
│   ├── evaluation/      # Evaluation scripts
│   ├── utils/           # Utility scripts
│   └── [other scripts]  # Other utilities
│
├── configs/              # Configuration files
│   ├── training_config.yaml
│   └── logging_config.yaml
│
├── docs/                  # Documentation
│   ├── README.md
│   ├── QUICK_START.md
│   └── *.md
│
├── thesis/                # Thesis LaTeX files
│   ├── thesis_main.tex
│   └── chap*/            # Chapter files
│
└── notebooks/             # Jupyter notebooks
    └── *.ipynb
```

---

## 📚 Documentation

### Quick Start
1. **Read Reports**: Start with `reports/REPORT_INDEX.md` for complete documentation
2. **Project Summary**: See `reports/general/PROJECT_SUMMARY.md`
3. **Model Reports**: 
   - Alpaca: `reports/alpaca/ALPACA_COMPLETE_REPORT.md`
   - QWEN: `reports/qwen/QWEN_COMPLETE_REPORT.md`
4. **Comparison**: `reports/comparison/COMPREHENSIVE_MODEL_COMPARISON_REPORT.md`

### Key Documents
- **Report Index**: `reports/REPORT_INDEX.md` - Complete guide to all reports
- **Project Summary**: `reports/general/PROJECT_SUMMARY.md` - Project overview
- **Fair Comparison**: `reports/comparison/FAIR_COMPARISON_METHODOLOGY_REPORT.md` - Methodology
- **Root Cause**: `reports/general/ROOT_CAUSE_ANALYSIS.md` - Analysis of limitations
- **Improvement Plan**: `reports/general/IMPROVEMENT_PLAN.md` - Future improvements

---

## 🔬 Models

### Alpaca-7B
- **Base Model**: LLaMA-7B (Alpaca)
- **Method**: QLoRA (4-bit quantization)
- **Epochs**: 3
- **Performance**: 40% accuracy, 46.15% F1 macro
- **Report**: `reports/alpaca/ALPACA_COMPLETE_REPORT.md`

### QWEN 2.5-7B
- **Base Model**: Qwen2.5-7B-Instruct
- **Method**: LoRA (full precision)
- **Epochs**: 5
- **Performance**: 50% accuracy, 16.44% F1 macro
- **Report**: `reports/qwen/QWEN_COMPLETE_REPORT.md`

### Llama3.1-8B
- **Status**: Planned
- **Directory**: `experiments/llama3.1-8b/`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (RTX 4090 tested)
- PyTorch
- Transformers library

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
bash setup.sh
```

### Training
```bash
# Train QWEN model
python finetune_qwen.py

# Train Alpaca model
python scripts/train_vetllm.py
```

### Evaluation
```bash
# Evaluate QWEN
python evaluate_qwen_comprehensive.py

# Compare models
python compare_models.py
```

---

## 📊 Results Summary

### Alpaca-7B
- ✅ Better rare disease handling
- ✅ More balanced F1 scores
- ✅ Memory efficient (4-bit quantization)

### QWEN 2.5-7B
- ✅ Higher overall accuracy
- ✅ Excellent on common diseases (PPR: 90.9%)
- ✅ Better validation loss

### Common Limitations
- ⚠️ Class imbalance affects rare disease performance
- ⚠️ SNOMED code accuracy needs improvement (33-35%)

---

## 🔧 Future Work

1. **Llama3.1-8B Training**: Fine-tune using same methodology
2. **Data Augmentation**: Generate examples for rare diseases
3. **Class-Weighted Training**: Address class imbalance
4. **Extended Evaluation**: Comprehensive three-way comparison

---

## 📝 Citation

If you use this work, please cite:
```
VetLLM: Fine-tuning Large Language Models for Veterinary Diagnosis Prediction
[Your citation details]
```

---

## 📄 License

[Your license information]

---

## 👥 Contributors

[Your contributors]

---

**Last Updated**: 2026-01-05  
**Status**: Core Research Complete ✅

