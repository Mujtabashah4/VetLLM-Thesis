# Llama 3.1 8B Setup and Training Guide

## 🔐 Step 1: Authenticate with HuggingFace

Since Llama 3.1 is a gated model, you need to authenticate first:

```bash
# Option 1: Using CLI
huggingface-cli login
# OR
hf auth login

# Option 2: Using Python
python -c "from huggingface_hub import login; login()"
```

You'll need:
1. A HuggingFace account
2. Access granted to Llama 3.1 (request at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. Your HuggingFace token (get it from: https://huggingface.co/settings/tokens)

---

## 📥 Step 2: Download the Model (Optional but Recommended)

Downloading the model first makes training faster:

```bash
python download_llama3.1.py
```

This will:
- Download the model to `models/llama3.1-8b-instruct/`
- Use 4-bit quantization to save memory
- Take ~10-30 minutes depending on internet speed

**Note**: If you skip this step, the model will be downloaded automatically during training, but it may be slower.

---

## 🚀 Step 3: Start Training

Once authenticated (and optionally downloaded), start training:

```bash
python start_training_llama3.1.py
```

Or use the experiment script:

```bash
cd experiments/llama3.1-8b
bash run_experiment.sh train
```

---

## 📋 Training Configuration

The training uses the following configuration (from `experiments/llama3.1-8b/configs/training_config.yaml`):

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Method**: QLoRA (4-bit quantization)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Epochs**: 3
- **Batch Size**: 4 per device × 4 gradient accumulation = 16 effective
- **Learning Rate**: 1e-4
- **Optimizer**: paged_adamw_8bit
- **Sequence Length**: 512
- **Training Data**: 373 samples
- **Validation Data**: 80 samples

---

## 📊 Expected Training Time

- **Per Epoch**: ~10-15 minutes (on RTX 4090)
- **Total (3 epochs)**: ~30-45 minutes
- **Model Size**: ~16 GB (with 4-bit quantization)

---

## 📁 Output Locations

After training completes:

- **Model Checkpoints**: `experiments/llama3.1-8b/checkpoints/`
- **Final Model**: `experiments/llama3.1-8b/checkpoints/final/`
- **Training Logs**: `experiments/llama3.1-8b/logs/`
- **Training Metrics**: `experiments/llama3.1-8b/checkpoints/final/training_metrics.json`

---

## 🔧 Troubleshooting

### Authentication Issues

If you get "401 Unauthorized" errors:

1. Check if you're logged in:
   ```bash
   huggingface-cli whoami
   ```

2. If not logged in:
   ```bash
   huggingface-cli login
   ```

3. Verify you have access to the model:
   - Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - Make sure you've accepted the license
   - Request access if needed

### Out of Memory (OOM)

If you get CUDA OOM errors:

1. The config already uses 4-bit quantization (QLoRA)
2. Try reducing batch size in `training_config.yaml`:
   - `per_device_train_batch_size: 2` (from 4)
   - `gradient_accumulation_steps: 8` (from 4)

### Model Download Issues

If model download fails:

1. Check internet connection
2. Ensure you have ~20 GB free disk space
3. Try downloading again (resumes automatically)
4. Check HuggingFace status: https://status.huggingface.co/

---

## ✅ Verification

After training starts, you should see:

1. Model loading messages
2. LoRA configuration
3. Trainable parameters count (~40M parameters)
4. Training progress with loss values
5. Validation loss after each evaluation

---

## 📝 Next Steps After Training

1. **Evaluate the model**: Use the evaluation script
2. **Compare with other models**: Generate comparison reports
3. **Update reports**: Add results to `reports/llama3.1/`

---

**Ready to start?** Run: `python start_training_llama3.1.py`

