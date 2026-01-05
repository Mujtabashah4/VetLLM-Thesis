#!/bin/bash
# Monitor Improved Training Progress

LOG_FILE="training_improved.log"

echo "=========================================="
echo "Monitoring Improved Training Progress"
echo "=========================================="
echo ""

# Check if training is running
if pgrep -f "train.py.*training_config_improved" > /dev/null; then
    echo "✅ Training is RUNNING"
else
    echo "⚠️  Training process not found"
fi

echo ""
echo "Latest Training Progress:"
echo "----------------------------------------"
tail -30 "$LOG_FILE" | grep -E "(epoch|loss|eval_loss|step|Best|Early)" | tail -15

echo ""
echo "Current Step:"
tail -5 "$LOG_FILE" | grep -E "(\d+%|\d+/\d+)" | tail -1

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | awk '{printf "  Memory: %d MB | GPU: %d%% | Temp: %d°C\n", $1, $2, $3}'

echo ""
echo "To monitor continuously: tail -f $LOG_FILE"

