#!/bin/bash
# Show real-time training progress

cd /home/iml_admin/Desktop/VetLLM/VetLLM-Thesis

echo "=========================================="
echo "VetLLM Training Progress Monitor"
echo "=========================================="
echo ""

# Check if training is running
if ps aux | grep -q "[p]ython.*train_vetllm"; then
    echo "✅ Training Status: RUNNING"
    echo ""
    
    # Show latest log output
    if [ -f training_output.log ]; then
        echo "--- Latest Training Output ---"
        tail -50 training_output.log
    else
        echo "Log file not found. Checking process..."
        ps aux | grep "[p]ython.*train_vetllm"
    fi
    
    echo ""
    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F', ' '{printf "GPU: %s | Usage: %s%% | Memory: %s / %s\n", $1, $2, $3, $4}'
    
else
    echo "❌ Training Status: NOT RUNNING"
    echo ""
    echo "To start training, run:"
    echo "  cd /home/iml_admin/Desktop/VetLLM/VetLLM-Thesis"
    echo "  source venv/bin/activate"
    echo "  python3 scripts/train_vetllm.py --model-name wxjiao/alpaca-7b --data-path processed_data/all_processed_data.json --output-dir models/vetllm-finetuned --epochs 3 --batch-size 6 --learning-rate 2e-4 --no-wandb"
fi

echo ""
echo "To watch live progress: tail -f training_output.log"

