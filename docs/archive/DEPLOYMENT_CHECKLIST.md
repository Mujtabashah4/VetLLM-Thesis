# ✅ VetLLM Deployment Checklist

**Complete checklist for deploying VetLLM on your training machine**

---

## 📦 Pre-Deployment

### Files to Transfer

Copy the entire `VetLLM/` directory to your training machine. Ensure these files are present:

- [ ] `setup.sh` - Setup script
- [ ] `start_training.sh` - Training start script
- [ ] `requirements.txt` - Dependencies
- [ ] `scripts/train_vetllm.py` - Training script
- [ ] `scripts/inference.py` - Inference script
- [ ] `scripts/validate_data.py` - Data validation
- [ ] `processed_data/all_processed_data.json` - Training data
- [ ] All documentation files (optional but recommended)

---

## 🖥️ System Requirements Check

### Hardware

- [ ] **GPU:** NVIDIA GPU with 16GB+ VRAM
- [ ] **RAM:** 32GB+ system RAM
- [ ] **Storage:** 50GB+ free space
- [ ] **Network:** Internet connection (for model download)

### Software

- [ ] **OS:** Linux, macOS, or Windows with WSL2
- [ ] **Python:** 3.10 or 3.11 installed
- [ ] **CUDA:** 11.8 or 12.1+ (if using GPU)
- [ ] **pip:** Latest version

### Verification

```bash
# Check Python
python3 --version  # Should be 3.10+

# Check GPU (if available)
nvidia-smi  # Should show GPU info

# Check CUDA (if using GPU)
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 🚀 Deployment Steps

### Step 1: Install Dependencies

```bash
cd VetLLM
chmod +x setup.sh
./setup.sh
```

**Checklist:**
- [ ] Setup script runs without errors
- [ ] All packages installed successfully
- [ ] GPU detected (if available)
- [ ] Data validation passed

### Step 2: Verify Data

```bash
python scripts/validate_data.py
```

**Checklist:**
- [ ] All data files validated
- [ ] No critical errors
- [ ] Sample count matches expected

### Step 3: Start Training

```bash
chmod +x start_training.sh
./start_training.sh
```

**Checklist:**
- [ ] Training starts successfully
- [ ] Model downloads (first time only)
- [ ] Training progress visible
- [ ] No errors in first few steps

---

## 📊 During Training

### Monitor Progress

- [ ] Training loss decreasing
- [ ] Validation loss (if validation data provided)
- [ ] Checkpoints saving regularly
- [ ] GPU utilization high (if using GPU)
- [ ] No out-of-memory errors

### Check Logs

```bash
# Watch training logs
tail -f models/vetllm-finetuned/logs/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## ✅ Post-Training Verification

### Check Output

- [ ] `models/vetllm-finetuned/` directory created
- [ ] `adapter_model.safetensors` exists (~20MB)
- [ ] `adapter_config.json` exists
- [ ] Checkpoints saved
- [ ] Training logs present

### Test Inference

```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --note "Cow. Clinical presentation includes epistaxis."
```

**Checklist:**
- [ ] Inference runs without errors
- [ ] Prediction generated
- [ ] SNOMED codes extracted (if using --extract-codes)
- [ ] Output format correct

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Out of Memory** | Reduce batch size: `--batch-size 2` |
| **CUDA not found** | Check CUDA installation: `nvidia-smi` |
| **Slow training** | Verify GPU is being used |
| **Model not loading** | Check internet connection for model download |
| **Data not found** | Verify `processed_data/` directory exists |

### Verification Commands

```bash
# Check GPU
nvidia-smi

# Check CUDA in Python
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check data files
ls -lh processed_data/*.json

# Check model directory
ls -lh models/vetllm-finetuned/
```

---

## 📋 Configuration Summary

### Current Settings (Full Precision)

```yaml
Model: wxjiao/alpaca-7b
Method: LoRA
Precision: FP16 (full precision, no quantization)
Batch Size: 4
Gradient Accumulation: 4
Effective Batch: 16
Learning Rate: 2e-4
Epochs: 3
LoRA Rank: 16
LoRA Alpha: 32
```

### Data Files

- `all_processed_data.json` - 1,602 samples ✅
- `Cow_Buffalo_processed.json` - 746 samples ✅
- `Sheep_Goat_processed.json` - 856 samples ✅

---

## 🎯 Success Criteria

Training is successful if:

- [ ] Training completes without errors
- [ ] Final model saved to `models/vetllm-finetuned/`
- [ ] Training loss decreases over time
- [ ] Inference produces reasonable predictions
- [ ] SNOMED codes extracted correctly

---

## 📚 Documentation Reference

- **Quick Start:** `START_HERE.md`
- **Full Guide:** `DEPLOYMENT_GUIDE.md`
- **Technical Details:** `PIPELINE_IMPLEMENTATION_REPORT.md`
- **Data Info:** `DATA_VALIDATION_SUMMARY.md`

---

## ✅ Final Checklist

Before considering deployment complete:

- [ ] All dependencies installed
- [ ] Data validated
- [ ] Training completed successfully
- [ ] Model saved correctly
- [ ] Inference tested and working
- [ ] Documentation reviewed

---

**Status:** ✅ Ready for Deployment  
**Last Updated:** December 2024

