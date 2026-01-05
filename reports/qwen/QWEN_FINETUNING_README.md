# QWEN Model Fine-tuning Guide

## ✅ Setup Complete!

Your QWEN fine-tuning setup is ready. All data has been validated and the model is downloaded.

## 📊 Data Validation Results

### Dataset Statistics
- **Training samples**: 373 (70.0%)
- **Validation samples**: 80 (15.0%)
- **Test samples**: 80 (15.0%)
- **Total samples**: 533

### Data Quality
- ✅ **100% SNOMED code coverage** across all splits
- ✅ **All samples have required fields** (text, input, output, metadata)
- ✅ **No validation errors** found
- ✅ **Good disease distribution** (22 different diseases)
- ✅ **Balanced animal distribution** (Cow, Buffalo, Goat, Sheep)

### Top Diseases in Training Set
1. Peste des Petits Ruminants: 122 samples
2. Foot and Mouth Disease: 56 samples
3. Mastitis: 48 samples
4. Hemorrhagic Septicemia: 42 samples
5. Black Quarter: 29 samples

## 🚀 How to Start Fine-tuning

### Option 1: Interactive Workflow (Recommended)
Run the complete workflow script that validates data and asks for confirmation:

```bash
python finetune_qwen.py
```

This script will:
1. ✅ Check if the model exists
2. ✅ Validate all data files
3. ✅ Show data summary
4. ⏸️ Wait for your confirmation
5. 🚀 Start training

### Option 2: Direct Training
If you want to start training directly:

```bash
python experiments/shared/train.py --config experiments/qwen2.5-7b/configs/training_config.yaml
```

### Option 3: Using the Experiment Script
You can also use the experiment runner:

```bash
cd experiments/qwen2.5-7b
bash run_experiment.sh train
```

## 📋 Training Configuration

- **Model**: Qwen2.5-7B-Instruct
- **Method**: QLoRA (4-bit quantization)
- **Epochs**: 3
- **Batch size**: 4 per device
- **Gradient accumulation**: 4 (effective batch size: 16)
- **Learning rate**: 1e-4
- **Optimizer**: paged_adamw_8bit
- **Output directory**: `experiments/qwen2.5-7b/checkpoints`

## 📁 Important Files

- **Training script**: `experiments/shared/train.py`
- **Config file**: `experiments/qwen2.5-7b/configs/training_config.yaml`
- **Training data**: `experiments/qwen2.5-7b/data/train.json`
- **Validation data**: `experiments/qwen2.5-7b/data/validation.json`
- **Test data**: `experiments/qwen2.5-7b/data/test.json`
- **Model location**: `models/qwen2.5-7b-instruct/`

## 🔍 Validation Scripts

### Validate Data Only
```bash
python validate_qwen_data.py
```

This will validate all data files and generate a report in `reports/qwen_data_validation.json`

## 📈 Monitoring Training

Training logs will be saved to:
- `experiments/qwen2.5-7b/logs/training_YYYYMMDD_HHMMSS.log`

Checkpoints will be saved to:
- `experiments/qwen2.5-7b/checkpoints/`

## ✅ Pre-flight Checklist

Before starting training, ensure:

- [x] Model downloaded (`models/qwen2.5-7b-instruct/`)
- [x] Data validated (all files pass validation)
- [x] Training config exists (`experiments/qwen2.5-7b/configs/training_config.yaml`)
- [x] Sufficient GPU memory (recommended: 16GB+ VRAM)
- [ ] CUDA available (check with `nvidia-smi`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)

## 🎯 Next Steps After Training

1. **Evaluate the model**:
   ```bash
   python comprehensive_validation.py
   ```

2. **Test inference**:
   ```bash
   python experiments/shared/inference.py --model-path experiments/qwen2.5-7b/checkpoints/final
   ```

3. **Compare with baseline**:
   Check the evaluation results in `experiments/qwen2.5-7b/results/`

## 📝 Notes

- Training time depends on your hardware (typically 2-6 hours on a modern GPU)
- The model uses QLoRA for efficient fine-tuning (only ~1% of parameters are trainable)
- All checkpoints are saved automatically during training
- Best model (lowest validation loss) is loaded automatically at the end

## 🆘 Troubleshooting

If you encounter issues:

1. **Out of memory**: Reduce `per_device_train_batch_size` in config
2. **CUDA errors**: Check GPU availability with `nvidia-smi`
3. **Import errors**: Install dependencies: `pip install -r requirements.txt`
4. **Data errors**: Re-run validation: `python validate_qwen_data.py`

---

**Ready to start?** Run: `python finetune_qwen.py`

