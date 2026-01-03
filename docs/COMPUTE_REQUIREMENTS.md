# VetLLM Compute Requirements & Token Information

**Complete guide to compute requirements, token lengths, and context settings**

---

##  Current Configuration

### Token & Context Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Max Sequence Length** | **512 tokens** | Maximum input + output tokens per sample |
| **Max New Tokens (Inference)** | 100 tokens | Maximum tokens to generate during inference |
| **Context Window** | 512 tokens | Full context window for training |
| **Tokenization** | SentencePiece | LLaMA tokenizer (Alpaca-7B) |

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Per-Device Batch Size** | 4 | Samples processed per GPU per step |
| **Gradient Accumulation** | 4 steps | Effective batch size = 16 |
| **Effective Batch Size** | **16 samples** | Total samples per update |
| **Precision** | FP16 (mixed) | Full precision training (no quantization) |
| **Gradient Checkpointing** | Enabled | Memory optimization |

---

##  GPU Memory Requirements

### Full Precision Training (FP16)

**Base Model:** Alpaca-7B (7 billion parameters)

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **Model Weights (FP16)** | ~14 GB | 7B params × 2 bytes |
| **Optimizer States** | ~14 GB | AdamW optimizer |
| **Activations (Batch=4, Seq=512)** | ~4-6 GB | Forward pass activations |
| **Gradients** | ~14 GB | Model gradients |
| **LoRA Adapters** | ~0.1 GB | Only 0.4% parameters trainable |
| **Overhead** | ~2-4 GB | CUDA overhead, buffers |
| **TOTAL** | **~48-52 GB** | Peak memory usage |

### With Gradient Checkpointing

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **Model Weights (FP16)** | ~14 GB | |
| **Optimizer States** | ~14 GB | |
| **Activations (Checkpointed)** | ~2-3 GB | Reduced by ~50% |
| **Gradients** | ~14 GB | |
| **LoRA Adapters** | ~0.1 GB | |
| **Overhead** | ~2-4 GB | |
| **TOTAL** | **~46-49 GB** | With checkpointing |

### Minimum GPU Requirements

| GPU Model | VRAM | Status | Notes |
|-----------|------|--------|-------|
| **RTX 3090** | 24 GB | ️ **Tight** | May need batch size 2 |
| **RTX 4090** | 24 GB | ️ **Tight** | May need batch size 2 |
| **A100 (40GB)** | 40 GB |  **Recommended** | Comfortable margin |
| **A100 (80GB)** | 80 GB |  **Excellent** | Plenty of headroom |
| **V100 (32GB)** | 32 GB | ️ **Marginal** | May need optimization |

### Recommended GPU Setup

**Minimum:** NVIDIA GPU with **24GB+ VRAM**  
**Recommended:** NVIDIA GPU with **40GB+ VRAM** (A100)

---

## ⏱️ Training Time Estimates

### Per Epoch Time (1,602 samples)

| GPU Model | Time per Epoch | Total (3 epochs) | Notes |
|-----------|----------------|-------------------|-------|
| **A100 (40GB)** | ~30-45 min | ~2-3 hours | Optimal performance |
| **V100 (32GB)** | ~45-60 min | ~3-4 hours | Good performance |
| **RTX 3090 (24GB)** | ~60-90 min | ~4-6 hours | May need batch=2 |
| **RTX 4090 (24GB)** | ~45-60 min | ~3-4 hours | Good performance |

### Training Speed

- **Samples per Second:** ~0.5-1.5 (depends on GPU)
- **Steps per Epoch:** ~100 steps (1,602 samples ÷ 16 effective batch)
- **Total Steps (3 epochs):** ~300 steps

---

##  Token Length Analysis

### Average Token Counts

Based on your data structure:

| Component | Average Characters | Estimated Tokens | Notes |
|-----------|-------------------|-----------------|-------|
| **Instruction** | ~80 chars | ~20 tokens | Fixed format |
| **Input (Clinical Note)** | ~140 chars | ~35 tokens | Average from data |
| **Output (SNOMED codes)** | ~30 chars | ~8 tokens | Short responses |
| **Prompt Formatting** | ~150 chars | ~40 tokens | Alpaca template |
| **TOTAL per Sample** | ~400 chars | **~103 tokens** | Average |

### Token Distribution

- **Min Tokens:** ~60 tokens (short notes)
- **Max Tokens:** ~200 tokens (long notes)
- **Average:** ~103 tokens
- **Max Length Setting:** 512 tokens (plenty of headroom)

### Context Window Usage

- **Used:** ~103 tokens (average)
- **Available:** 512 tokens
- **Utilization:** ~20% (efficient)
- **Headroom:** ~409 tokens (can handle longer notes)

---

##  Data Size Calculations

### Dataset Size

| Dataset | Samples | Avg Tokens | Total Tokens | Storage |
|---------|---------|------------|--------------|---------|
| **All Processed** | 1,602 | ~103 | ~165K | ~2.5 MB |
| **Cow/Buffalo** | 746 | ~103 | ~77K | ~1.2 MB |
| **Sheep/Goat** | 856 | ~103 | ~88K | ~1.3 MB |

### Training Data Throughput

- **Samples per Epoch:** 1,602
- **Tokens per Epoch:** ~165,000 tokens
- **Total Tokens (3 epochs):** ~495,000 tokens
- **Training Steps:** ~300 steps (16 effective batch)

---

##  Memory Optimization Options

### Option 1: Reduce Batch Size

```bash
python scripts/train_vetllm.py --batch-size 2 ...
```

- Reduces memory by ~25-30%
- Increases training time by ~2x
- Still maintains good gradient estimates

### Option 2: Increase Gradient Accumulation

```bash
python scripts/train_vetllm.py --batch-size 2 --gradient-accumulation-steps 8 ...
```

- Effective batch size remains 16
- Reduces peak memory
- Slightly slower training

### Option 3: Enable 8-bit Quantization

Edit `scripts/train_vetllm.py` and set `use_8bit=True`

- Reduces memory by ~50% (to ~24-26 GB)
- May slightly reduce accuracy
- Enables training on 24GB GPUs

### Option 4: Reduce Sequence Length

Edit `scripts/train_vetllm.py` and set `max_length=256`

- Reduces memory by ~30-40%
- May truncate longer notes
- Not recommended (your notes fit in 512)

---

##  Scaling Considerations

### If You Have More Data

| Samples | Training Time (A100) | Memory Usage | Notes |
|---------|---------------------|--------------|-------|
| 1,602 (current) | ~2-3 hours | ~48 GB | Current setup |
| 5,000 | ~6-9 hours | ~48 GB | Same memory |
| 10,000 | ~12-18 hours | ~48 GB | Same memory |
| 20,000 | ~24-36 hours | ~48 GB | Same memory |

**Note:** Memory usage doesn't increase with more data (only training time).

### If You Need Longer Context

| Max Length | Memory Usage | Notes |
|------------|--------------|-------|
| 512 (current) | ~48 GB | Current setup |
| 1024 | ~56 GB | +17% memory |
| 2048 | ~72 GB | +50% memory |
| 4096 | ~104 GB | +117% memory |

**Note:** Your current data fits comfortably in 512 tokens.

---

## ️ System Requirements Summary

### Minimum Requirements

- **GPU:** NVIDIA 24GB+ VRAM (RTX 3090/4090)
- **RAM:** 32GB+ system RAM
- **Storage:** 50GB+ free space
- **CUDA:** 11.8+ or 12.1+

### Recommended Requirements

- **GPU:** NVIDIA 40GB+ VRAM (A100)
- **RAM:** 64GB+ system RAM
- **Storage:** 100GB+ free space (SSD recommended)
- **CUDA:** 12.1+

### Optimal Setup

- **GPU:** NVIDIA A100 (40GB or 80GB)
- **RAM:** 128GB+ system RAM
- **Storage:** 200GB+ NVMe SSD
- **CUDA:** 12.1+
- **Network:** High-speed for model download

---

##  Token Efficiency

### Your Data Characteristics

- **Average Note Length:** ~140 characters
- **Average Token Count:** ~103 tokens
- **Max Length Setting:** 512 tokens
- **Efficiency:** ~20% utilization (very efficient!)

### Why 512 Tokens is Optimal

1. **Sufficient Headroom:** 5x average length
2. **Memory Efficient:** Not too large
3. **Standard Size:** Common for instruction tuning
4. **No Truncation:** All your notes fit comfortably

---

##  Detailed Memory Breakdown

### Per-Sample Memory (Batch Size = 4)

| Component | Memory per Sample | Total (Batch=4) |
|-----------|------------------|-----------------|
| Input Tokens (512 max) | ~2 KB | ~8 KB |
| Attention Matrices | ~1 MB | ~4 MB |
| Hidden States | ~0.5 MB | ~2 MB |
| Gradients | ~14 GB (shared) | ~14 GB |
| **Total per Sample** | ~0.5 MB | **~2 MB** |

### Peak Memory During Training

```
Model Weights:        14 GB
Optimizer States:     14 GB
Gradients:           14 GB
Activations (batch=4):  4 GB
LoRA Adapters:         0.1 GB
CUDA Overhead:         2 GB
─────────────────────────────
TOTAL:               ~48 GB
```

---

##  Recommendations

### For Your Current Setup (1,602 samples)

1. **Use Default Settings:**
   - Batch size: 4
   - Max length: 512
   - FP16 precision
   - Gradient checkpointing

2. **Recommended GPU:**
   - Minimum: RTX 3090/4090 (24GB)
   - Recommended: A100 (40GB)

3. **Training Time:**
   - Expect: 2-4 hours on A100
   - Expect: 4-6 hours on RTX 3090

### If You Have Limited GPU Memory

1. **Reduce batch size to 2:**
   ```bash
   python scripts/train_vetllm.py --batch-size 2 ...
   ```

2. **Or enable 8-bit quantization:**
   - Edit `scripts/train_vetllm.py`
   - Set `use_8bit=True`

---

##  Summary

### Key Numbers

- **Context Length:** 512 tokens
- **Average Usage:** ~103 tokens (20% utilization)
- **GPU Memory:** ~48 GB (FP16, batch=4)
- **Training Time:** 2-4 hours (A100, 3 epochs)
- **Data Size:** 1,602 samples, ~165K tokens/epoch

### Your Data is Efficient

-  All notes fit in 512 tokens
-  No truncation needed
-  Plenty of headroom for longer notes
-  Optimal token utilization

---

