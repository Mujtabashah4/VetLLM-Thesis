# VetLLM - Production Deployment Guide

**Ready-to-deploy fine-tuning pipeline for Alpaca-7B on veterinary data**

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
./setup.sh

# 2. Validate data (optional)
python scripts/validate_data.py

# 3. Start training
./start_training.sh
```

**That's it!** Your model will be trained and saved to `models/vetllm-finetuned/`

---

## 📋 What You Get

After running the pipeline, you'll have:

✅ **Fine-tuned LoRA adapters** (~20MB)  
✅ **Training logs** and checkpoints  
✅ **Ready-to-use model** for inference  

---

## 🎯 Training Configuration

The pipeline uses **optimal settings** based on the working notebook:

- **Model:** Alpaca-7B (wxjiao/alpaca-7b)
- **Method:** LoRA fine-tuning
- **Precision:** FP16 mixed precision (full precision, no quantization)
- **Batch Size:** 4 per device
- **Effective Batch:** 16 (with gradient accumulation)
- **Learning Rate:** 2e-4 (optimized for LoRA)
- **Epochs:** 3 (configurable)

---

## 📁 Data Files

All data is **validated and ready**:

| File | Samples | Status |
|------|---------|--------|
| `all_processed_data.json` | 1,602 | ✅ Ready |
| `Cow_Buffalo_processed.json` | 746 | ✅ Ready |
| `Sheep_Goat_processed.json` | 856 | ✅ Ready |

**Total:** 3,204 validated samples

---

## 💻 System Requirements

### Minimum
- **GPU:** NVIDIA 16GB+ VRAM
- **RAM:** 32GB system RAM
- **Storage:** 50GB free space

### Recommended
- **GPU:** NVIDIA 24GB+ VRAM (A100, V100, RTX 3090/4090)
- **CUDA:** 11.8 or 12.1+
- **Python:** 3.10 or 3.11

---

## 📖 Usage

### Basic Training

```bash
python scripts/train_vetllm.py \
    --data-path processed_data/all_processed_data.json \
    --output-dir models/vetllm-finetuned \
    --epochs 3
```

### Using Start Script

```bash
# Default (uses all_processed_data.json, 3 epochs)
./start_training.sh

# Custom data file
./start_training.sh processed_data/Verified_DLO_data_-_\(Cow_Buffalo\)_processed.json

# Custom epochs
./start_training.sh processed_data/all_processed_data.json models/vetllm-finetuned 5
```

### Run Inference

```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --note "Cow. Clinical presentation includes epistaxis and high fever."
```

---

## ⚙️ Configuration

### Default Settings (Optimal)

These settings are optimized based on the working notebook:

```python
# LoRA Configuration
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1

# Training Configuration
batch_size = 4
gradient_accumulation_steps = 4  # Effective batch = 16
learning_rate = 2e-4  # Optimized for LoRA
epochs = 3
fp16 = True  # Mixed precision (full precision training)
gradient_checkpointing = True
```

### Customize Training

Edit `scripts/train_vetllm.py` or use command-line arguments:

```bash
python scripts/train_vetllm.py \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    ...
```

---

## 📊 Expected Results

### Training Time

| GPU | Time per Epoch | Total (3 epochs) |
|-----|----------------|------------------|
| A100 (40GB) | ~30-45 min | ~2-3 hours |
| V100 (32GB) | ~45-60 min | ~3-4 hours |
| RTX 3090 (24GB) | ~60-90 min | ~4-6 hours |

### Model Output

After training, you'll find:
- LoRA adapters in `models/vetllm-finetuned/`
- Training logs in `models/vetllm-finetuned/logs/`
- Checkpoints saved every 100 steps

---

## 🔧 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python scripts/train_vetllm.py --batch-size 2 ...
```

### Slow Training

1. Check CUDA is available:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. Monitor GPU usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

### Poor Results

- Train for more epochs: `--epochs 5`
- Use more data
- Adjust learning rate: `--learning-rate 1e-4`

---

## 📚 Documentation

- **Full Guide:** `DEPLOYMENT_GUIDE.md`
- **Implementation Report:** `PIPELINE_IMPLEMENTATION_REPORT.md`
- **Quick Start:** `QUICK_START.md`
- **Data Validation:** `DATA_VALIDATION_SUMMARY.md`

---

## ✅ Pre-Deployment Checklist

- [ ] GPU with 16GB+ VRAM available
- [ ] CUDA installed and working
- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`./setup.sh`)
- [ ] Data files validated (`python scripts/validate_data.py`)
- [ ] Sufficient disk space (50GB+)
- [ ] Output directory writable

---

## 🎓 Next Steps

1. **Train the model:**
   ```bash
   ./start_training.sh
   ```

2. **Test inference:**
   ```bash
   python scripts/inference.py \
       --model models/vetllm-finetuned \
       --base-model-name wxjiao/alpaca-7b \
       --note "Your test note"
   ```

3. **Evaluate performance:**
   - Test on validation set
   - Calculate metrics
   - Compare with baseline

---

**Ready to deploy!** Just run `./setup.sh` and `./start_training.sh` 🚀

