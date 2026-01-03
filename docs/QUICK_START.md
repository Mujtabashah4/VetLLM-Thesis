# VetLLM Quick Start Guide

**Get started in 3 simple steps**

---

## Step 1: Install Dependencies

```bash
chmod +x setup.sh
./setup.sh
```

This installs all required packages and validates your system.

---

## Step 2: Start Training

```bash
chmod +x start_training.sh
./start_training.sh
```

This starts fine-tuning with optimal settings.

---

## Step 3: Run Inference

```bash
python scripts/inference.py \
    --model models/vetllm-finetuned \
    --base-model-name wxjiao/alpaca-7b \
    --note "Cow. Clinical presentation includes epistaxis."
```

---

## Configuration

- Model: Alpaca-7B (wxjiao/alpaca-7b)
- Method: LoRA fine-tuning
- Precision: FP16 (full precision)
- Batch Size: 4 per device
- Learning Rate: 2e-4
- Epochs: 3

- `processed_data/all_processed_data.json` - 1,602 samples
- `processed_data/Verified_DLO_data_-_(Cow_Buffalo)_processed.json` - 746 samples
- `processed_data/Verified_DLO_data_(Sheep_Goat)_processed.json` - 856 samples

---

## System Requirements

- **GPU:** NVIDIA 16GB+ VRAM
- **RAM:** 32GB+ system RAM
- **Storage:** 50GB+ free space
- **Python:** 3.10+

---

## Troubleshooting

**Out of Memory?**
```bash
python scripts/train_vetllm.py --batch-size 2 ...
```

**Slow Training?**
- Check CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Monitor GPU: `watch -n 1 nvidia-smi`

---

For detailed instructions, see [Deployment Guide](DEPLOYMENT_GUIDE.md)

