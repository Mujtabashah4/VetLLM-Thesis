# VetLLM - Veterinary Large Language Model

**Fine-tuning Alpaca-7B for Veterinary Diagnosis Prediction using SNOMED-CT Codes**

---

##  Quick Start

### 1. Install Dependencies
```bash
./setup.sh
```

### 2. Start Training
```bash
./start_training.sh
```

### 3. Run Inference
```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --note "Cow. Clinical presentation includes epistaxis."
```

**That's it!** See [docs/QUICK_START.md](docs/QUICK_START.md) for detailed instructions.

---

##  Overview

VetLLM fine-tunes Alpaca-7B (LLaMA-2 based) using LoRA on validated veterinary clinical data to predict SNOMED-CT diagnosis codes.

### Key Features

-  **Production-Ready Pipeline** - Just install and run
-  **Full Precision Training** - FP16 mixed precision (optimized for accuracy)
-  **Validated Data** - 3,204 samples ready for training
-  **Comprehensive Documentation** - Complete guides in `docs/`
-  **Automated Setup** - One-command installation

---

##  Project Structure

```
VetLLM/
├── README.md                    #  This file (START HERE)
├── setup.sh                     # Automated setup
├── start_training.sh            # Training start script
├── requirements.txt            # Dependencies
│
├── docs/                        #  All Documentation
│   ├── README.md               # Documentation index
│   ├── QUICK_START.md          # Quick start guide
│   ├── DEPLOYMENT_GUIDE.md     # Complete deployment guide
│   ├── IMPLEMENTATION_REPORT.md # Technical details
│   ├── DATA_VALIDATION.md      # Data validation results
│   └── DISSERTATION_REPORT.md  #  Complete report for defense
│
├── scripts/                     #  Core Scripts
│   ├── train_vetllm.py         # Main training script
│   ├── inference.py            # Inference script
│   ├── validate_data.py        # Data validation
│   └── run_pipeline.py         # Pipeline orchestrator
│
├── processed_data/              #  Training Data (Validated)
│   ├── all_processed_data.json # 1,602 samples (RECOMMENDED)
│   ├── Verified_DLO_data_-_(Cow_Buffalo)_processed.json
│   └── Verified_DLO_data_(Sheep_Goat)_processed.json
│
├── configs/                     # ️ Configuration Files
└── models/                      #  Trained Models (created during training)
```

**See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for complete structure guide.**

---

##  Training Configuration

- **Model:** Alpaca-7B (wxjiao/alpaca-7b)
- **Method:** LoRA fine-tuning
- **Precision:** FP16 mixed precision (full precision)
- **Batch Size:** 4 per device
- **Effective Batch:** 16 (gradient accumulation)
- **Learning Rate:** 2e-4 (optimized for LoRA)
- **Epochs:** 3 (configurable)

---

##  Data

All data files are **validated and ready**:

| File | Samples | SNOMED Coverage | Status |
|------|---------|----------------|--------|
| `all_processed_data.json` | 1,602 | 97.4% |  Ready |
| `Cow_Buffalo_processed.json` | 746 | 100% |  Ready |
| `Sheep_Goat_processed.json` | 856 | 95.1% |  Ready |

**Total:** 3,204 validated samples

---

##  System Requirements

- **GPU:** NVIDIA GPU with 16GB+ VRAM (recommended: 24GB+)
- **RAM:** 32GB+ system RAM
- **Storage:** 50GB+ free space
- **Python:** 3.10+
- **CUDA:** 11.8+ (for GPU)

---

##  Documentation

All documentation is in the `docs/` directory:

- **[Quick Start](docs/QUICK_START.md)** - Get started in 3 steps
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[Implementation Report](docs/IMPLEMENTATION_REPORT.md)** - Technical details
- **[Data Validation](docs/DATA_VALIDATION.md)** - Data validation results
- **[Dissertation Report](docs/DISSERTATION_REPORT.md)** -  Complete report for defense

---

##  Usage Examples

### Basic Training
```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 3
```

### Inference
```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --note "Cow. Clinical presentation includes epistaxis."
```

---

##  Status

-  **Data Validated** - All files ready
-  **Pipeline Complete** - 
-  **Documentation Complete** - All guides in `docs/`
-  **Optimized Configuration** - Based on working notebook
-  **Full Precision Training** - No quantization by default
-  **Directory Organized** - Clean structure, easy navigation

---

##  For Dissertation Defense

See **[docs/DISSERTATION_REPORT.md](docs/DISSERTATION_REPORT.md)** for the complete implementation report suitable for dissertation defense.

---

