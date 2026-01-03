#!/bin/bash
# VetLLM Training Start Script
# Simple script to start fine-tuning with optimal settings

set -e  # Exit on error

echo "=========================================="
echo "VetLLM Fine-Tuning"
echo "=========================================="
echo ""

# Default configuration
DATA_PATH="${1:-processed_data/all_processed_data.json}"
OUTPUT_DIR="${2:-models/vetllm-finetuned}"
EPOCHS="${3:-3}"

echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo ""

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    echo ""
    echo "Available data files:"
    ls -1 processed_data/*.json 2>/dev/null || echo "  No data files found in processed_data/"
    exit 1
fi

# Check if CUDA is available
python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ CUDA available - Using GPU for training"
    DEVICE_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "   GPU: $DEVICE_INFO"
else
    echo "⚠️  CUDA not available - Training will be slow on CPU"
fi

echo ""
echo "Starting training..."
echo ""

# Start training with optimal settings
python3 scripts/train_vetllm.py \
    --model-name wxjiao/alpaca-7b \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size 4 \
    --learning-rate 2e-4

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "You can now run inference with:"
echo "  python scripts/inference.py --model $OUTPUT_DIR --base-model-name wxjiao/alpaca-7b --note 'Your clinical note here'"
echo ""

