# VetLLM Deployment Guide

**Complete guide for deploying and running VetLLM fine-tuning pipeline**

---

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
pip install -r requirements.txt
```

### Step 2: Validate Data (Optional but Recommended)

```bash
python scripts/validate_data.py
```

### Step 3: Start Training

```bash
chmod +x start_training.sh
./start_training.sh
```

Or manually:
```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 3
```

**That's it!** The model will start training and save to `models/vetllm-finetuned/`

---

## System Requirements

### Minimum Requirements

- **GPU:** NVIDIA GPU with 16GB+ VRAM (recommended: 24GB+)
- **CPU:** Multi-core CPU (8+ cores recommended)
- **RAM:** 32GB+ system RAM
- **Storage:** 50GB+ free space
- **OS:** Linux, macOS, or Windows with WSL2

### Recommended Setup

- **GPU:** NVIDIA A100, V100, or RTX 3090/4090
- **CUDA:** 11.8 or 12.1+
- **Python:** 3.10 or 3.11
- **Storage:** SSD recommended for faster I/O

---

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Make scripts executable
chmod +x setup.sh start_training.sh

# Run setup
./setup.sh
```

### Option 2: Manual Installation

```bash
# Create virtual environment (recommended)
python3 -m venv vetllm_env
source vetllm_env/bin/activate  # Linux/Mac
# or
vetllm_env\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Training Configuration

### Default Settings (Optimized for Full Precision)

Based on the working notebook, these are the optimal settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | wxjiao/alpaca-7b | Alpaca-7B (LLaMA-2 based) |
| **Method** | LoRA | Low-Rank Adaptation |
| **LoRA Rank** | 16 | LoRA rank |
| **LoRA Alpha** | 32 | LoRA scaling factor |
| **Batch Size** | 4 | Per-device batch size |
| **Gradient Accumulation** | 4 | Effective batch = 16 |
| **Learning Rate** | 2e-4 | Optimized for LoRA |
| **Epochs** | 3 | Number of training epochs |
| **Precision** | FP16 | Mixed precision (CUDA) |
| **Gradient Checkpointing** | Yes | Memory optimization |

### Training Time Estimates

| GPU | Time per Epoch | Total (3 epochs) |
|-----|----------------|------------------|
| A100 (40GB) | ~30-45 min | ~2-3 hours |
| V100 (32GB) | ~45-60 min | ~3-4 hours |
| RTX 3090 (24GB) | ~60-90 min | ~4-6 hours |
| RTX 4090 (24GB) | ~45-60 min | ~3-4 hours |

---

## Usage Examples

### Basic Training

```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 3
```

### Training with Validation Data

```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --val-data-path data/processed/val_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 3
```

### Custom Configuration

```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 1e-4
```

### Using Start Script

```bash
# Default settings
./start_training.sh

# Custom data file
./start_training.sh processed_data/Verified_DLO_data_-_\(Cow_Buffalo\)_processed.json

# Custom output directory
./start_training.sh processed_data/all_processed_data.json models/my-model

# Custom epochs
./start_training.sh processed_data/all_processed_data.json models/vetllm-finetuned 5
```

---

## Running Inference

### After Training

```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --note "Cow. Clinical presentation includes epistaxis and high fever."
```

### Batch Inference

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

## Monitoring Training

### Check Training Progress

Training logs are saved to `models/vetllm-finetuned/logs/`. You can monitor:

```bash
# Watch logs in real-time
tail -f models/vetllm-finetuned/logs/training.log

# Check latest checkpoint
ls -lh models/vetllm-finetuned/checkpoint-*/
```

### Training Output

The script will display:
- Training loss at each step
- Validation loss (if validation data provided)
- Training speed (samples/second)
- Estimated time remaining

### Checkpoints

Checkpoints are saved every 100 steps (configurable). The best model is automatically loaded at the end.

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

1. Reduce batch size:
   ```bash
   python scripts/train_vetllm.py --batch-size 2 ...
   ```

2. Increase gradient accumulation:
   ```bash
   python scripts/train_vetllm.py --batch-size 2 --gradient-accumulation-steps 8 ...
   ```

3. Enable 8-bit quantization (if needed):
   ```bash
   # Edit scripts/train_vetllm.py and set use_8bit=True
   ```

### Slow Training

1. Ensure CUDA is available:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. Check GPU utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. Use FP16 (default, already enabled)

### Model Not Saving

- Disk space available
- Write permissions on output directory
- Check logs for errors

### Poor Results

1. Train for more epochs:
   ```bash
   python scripts/train_vetllm.py --epochs 5 ...
   ```

2. Use more training data
3. Adjust learning rate:
   ```bash
   python scripts/train_vetllm.py --learning-rate 1e-4 ...
   ```

---

## Data Files

### Available Training Data

| File | Samples | SNOMED Coverage | Use Case |
|------|---------|-----------------|----------|
| `all_processed_data.json` | 1,602 | 97.4% | **Recommended** - All data |
| `Verified_DLO_data_-_(Cow_Buffalo)_processed.json` | 746 | 100% | Cow/Buffalo only |
| `Verified_DLO_data_(Sheep_Goat)_processed.json` | 856 | 95.1% | Sheep/Goat only |

All files are validated and ready to use!

---

## File Structure After Training

```
models/vetllm-finetuned/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights
├── checkpoint-100/              # Checkpoint at step 100
├── checkpoint-200/              # Checkpoint at step 200
├── ...
├── checkpoint-*/                # Latest checkpoint
├── logs/                        # Training logs
│   └── training.log
└── training_args.json           # Training arguments
```

**Note:** Only LoRA adapters are saved (~20MB), not the full model. The base model (`wxjiao/alpaca-7b`) must be available for inference.

---

## Next Steps After Training

1. **Test the Model:**
   ```bash
   python scripts/inference.py \
       --model models/vetllm-finetuned \
       --base-model-name wxjiao/alpaca-7b \
       --note "Your test clinical note"
   ```

2. **Evaluate Performance:**
   - Test on validation set
   - Calculate accuracy metrics
   - Compare with baseline

3. **Deploy:**
   - Create API wrapper
   - Deploy to production
   - Monitor performance

---

## Advanced Configuration

### Using Config File

Edit `configs/training_config.yaml` and use:
```bash
python scripts/train_vetllm.py --config configs/training_config.yaml
```

### Multi-GPU Training

```bash
# Using DeepSpeed
python scripts/train_vetllm.py \
    --deepspeed configs/deepspeed_config.json \
    ...
```

### Weights & Biases Logging

```bash
python scripts/train_vetllm.py \
    --wandb \
    ...
```

---

## Support

For issues or questions:
1. Check logs in `models/vetllm-finetuned/logs/`
2. Review `PIPELINE_IMPLEMENTATION_REPORT.md`
3. Check `TROUBLESHOOTING.md` (if available)

---

**
