# VetLLM Directory Structure

**Clean, organized structure for easy navigation**

---

## 📁 Main Directory Structure

```
VetLLM/
├── README.md                    # ⭐ Main project overview (START HERE)
├── setup.sh                     # Automated setup script
├── start_training.sh            # Training start script
├── requirements.txt            # Python dependencies
│
├── docs/                        # 📚 All Documentation
│   ├── README.md               # Documentation index
│   ├── QUICK_START.md          # Quick start guide
│   ├── DEPLOYMENT_GUIDE.md     # Complete deployment guide
│   ├── IMPLEMENTATION_REPORT.md # Technical implementation
│   ├── DATA_VALIDATION.md      # Data validation results
│   ├── DISSERTATION_REPORT.md  # Complete report for defense
│   └── archive/                 # Archived files
│       └── old_files/           # Old/redundant documentation
│
├── scripts/                     # 🔧 Core Scripts
│   ├── train_vetllm.py         # Main training script
│   ├── inference.py            # Inference script
│   ├── validate_data.py       # Data validation
│   ├── run_pipeline.py         # Pipeline orchestrator
│   ├── data_validation_report.py
│   └── test_data_loading.py
│
├── processed_data/              # 📊 Training Data (Validated)
│   ├── all_processed_data.json # 1,602 samples (RECOMMENDED)
│   ├── Verified_DLO_data_-_(Cow_Buffalo)_processed.json
│   └── Verified_DLO_data_(Sheep_Goat)_processed.json
│
├── configs/                     # ⚙️ Configuration Files
│   ├── training_config.yaml
│   ├── deepspeed_config.json
│   └── logging_config.yaml
│
├── models/                      # 🤖 Trained Models (Created during training)
│   ├── alpaca-7b/             # Base model cache
│   └── vetllm-finetuned/       # Fine-tuned model output
│
├── data/                        # 📁 Additional Data
│   ├── processed/              # Processed data splits
│   └── veterinary_notes/        # Veterinary notes
│
├── notebooks/                   # 📓 Jupyter Notebooks
│   └── VetLLM_Testing_Notebook.ipynb
│
├── thesis/                      # 📝 Thesis LaTeX Files
│   ├── thesis_main.tex
│   └── [chapters]
│
└── _backup_YYYYMMDD/           # 💾 Backup directory (created automatically)
```

---

## 🎯 Key Files

### Essential Files (Root Directory)

| File | Purpose |
|------|---------|
| `README.md` | Main project overview |
| `setup.sh` | Automated setup |
| `start_training.sh` | Start training |
| `requirements.txt` | Dependencies |

### Documentation (docs/)

| File | Purpose |
|------|---------|
| `QUICK_START.md` | Quick start guide |
| `DEPLOYMENT_GUIDE.md` | Complete deployment guide |
| `IMPLEMENTATION_REPORT.md` | Technical details |
| `DATA_VALIDATION.md` | Data validation results |
| `DISSERTATION_REPORT.md` | **Complete report for defense** |

### Scripts (scripts/)

| File | Purpose |
|------|---------|
| `train_vetllm.py` | **Main training script** |
| `inference.py` | **Inference script** |
| `validate_data.py` | Data validation |
| `run_pipeline.py` | Pipeline orchestrator |

### Data (processed_data/)

| File | Purpose |
|------|---------|
| `all_processed_data.json` | **Recommended training data** |
| `Cow_Buffalo_processed.json` | Cow/Buffalo specific |
| `Sheep_Goat_processed.json` | Sheep/Goat specific |

---

## 🚀 Quick Navigation

### To Start Training
1. Read: `README.md` or `docs/QUICK_START.md`
2. Run: `./setup.sh`
3. Train: `./start_training.sh`

### For Documentation
- All docs in: `docs/` directory
- Main guide: `docs/DEPLOYMENT_GUIDE.md`
- Defense report: `docs/DISSERTATION_REPORT.md`

### For Scripts
- All scripts in: `scripts/` directory
- Main training: `scripts/train_vetllm.py`
- Inference: `scripts/inference.py`

---

## 📋 File Organization Principles

1. **Root Directory:** Only essential files
2. **docs/:** All documentation organized
3. **scripts/:** All executable scripts
4. **processed_data/:** Validated training data
5. **models/:** Trained models (created during training)
6. **archive/:** Old/redundant files

---

**Last Updated:** December 2024

