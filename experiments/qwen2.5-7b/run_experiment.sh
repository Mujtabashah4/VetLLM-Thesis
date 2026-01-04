#!/bin/bash
# VetLLM Experiment Runner - Qwen2.5 7B
# Secondary comparison model (reasoning-focused)

set -e

# ============================================
# Configuration
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$EXPERIMENTS_DIR")"
SHARED_DIR="$EXPERIMENTS_DIR/shared"

MODEL_DIR="$SCRIPT_DIR"
DATA_DIR="$MODEL_DIR/data"
CONFIGS_DIR="$MODEL_DIR/configs"
CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"
RESULTS_DIR="$MODEL_DIR/results"
LOGS_DIR="$MODEL_DIR/logs"

# Model settings
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_TYPE="qwen2.5"
EXPERIMENT_NAME="qwen2.5-7b-vetllm"

# ============================================
# Helper Functions
# ============================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "GPU Status:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    else
        log "Warning: nvidia-smi not found. GPU may not be available."
    fi
}

# ============================================
# Step 1: Data Preprocessing
# ============================================
preprocess_data() {
    log "Step 1: Preprocessing data for Qwen2.5..."
    
    mkdir -p "$DATA_DIR"
    
    python "$SHARED_DIR/data_preprocessor.py" \
        --dataset-dir "$PROJECT_ROOT/Dataset_UVAS" \
        --output-dir "$DATA_DIR" \
        --model "$MODEL_TYPE" \
        --seed 42
    
    log "Data preprocessing complete. Output in: $DATA_DIR"
}

# ============================================
# Step 2: Training
# ============================================
train_model() {
    log "Step 2: Training Qwen2.5 7B..."
    
    mkdir -p "$CHECKPOINTS_DIR" "$LOGS_DIR"
    
    # Set environment variables for optimal training
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export TOKENIZERS_PARALLELISM=false
    
    python "$SHARED_DIR/train.py" \
        --config "$CONFIGS_DIR/training_config.yaml" \
        2>&1 | tee "$LOGS_DIR/training_$(date +%Y%m%d_%H%M%S).log"
    
    log "Training complete. Checkpoints in: $CHECKPOINTS_DIR"
}

# ============================================
# Step 3: Evaluation
# ============================================
evaluate_model() {
    log "Step 3: Evaluating model..."
    
    mkdir -p "$RESULTS_DIR"
    
    # Evaluate fine-tuned model
    python "$SHARED_DIR/evaluation/evaluate.py" \
        --model-path "$CHECKPOINTS_DIR/final" \
        --base-model "$MODEL_NAME" \
        --test-data "$DATA_DIR/test.json" \
        --output-dir "$RESULTS_DIR/finetuned" \
        2>&1 | tee "$LOGS_DIR/eval_finetuned_$(date +%Y%m%d_%H%M%S).log"
    
    # Evaluate base model (zero-shot baseline)
    log "Evaluating base model (zero-shot)..."
    python "$SHARED_DIR/evaluation/evaluate.py" \
        --model-path "$MODEL_NAME" \
        --test-data "$DATA_DIR/test.json" \
        --output-dir "$RESULTS_DIR/baseline" \
        --no-adapter \
        --num-samples 50 \
        2>&1 | tee "$LOGS_DIR/eval_baseline_$(date +%Y%m%d_%H%M%S).log"
    
    log "Evaluation complete. Results in: $RESULTS_DIR"
}

# ============================================
# Step 4: Generate Comparison Report
# ============================================
generate_report() {
    log "Step 4: Generating comparison report..."
    
    # Create a simple comparison summary
    cat > "$RESULTS_DIR/comparison_summary.md" << EOF
# VetLLM Experiment Results: Qwen2.5 7B

## Experiment: $EXPERIMENT_NAME
Date: $(date '+%Y-%m-%d %H:%M:%S')

## Model Information
- Base Model: $MODEL_NAME
- Fine-tuning: QLoRA (4-bit quantization)
- Training epochs: 3

## Results Summary

### Baseline (Zero-shot)
$(cat "$RESULTS_DIR/baseline/evaluation_summary.json" 2>/dev/null || echo "Results not available")

### Fine-tuned
$(cat "$RESULTS_DIR/finetuned/evaluation_summary.json" 2>/dev/null || echo "Results not available")

## Conclusion
See detailed results in the respective directories.
EOF
    
    log "Report generated: $RESULTS_DIR/comparison_summary.md"
}

# ============================================
# Main Execution
# ============================================
main() {
    log "============================================"
    log "VetLLM Experiment: Qwen2.5 7B"
    log "============================================"
    
    check_gpu
    
    case "${1:-all}" in
        preprocess)
            preprocess_data
            ;;
        train)
            train_model
            ;;
        evaluate)
            evaluate_model
            ;;
        report)
            generate_report
            ;;
        all)
            preprocess_data
            train_model
            evaluate_model
            generate_report
            ;;
        *)
            echo "Usage: $0 {preprocess|train|evaluate|report|all}"
            exit 1
            ;;
    esac
    
    log "============================================"
    log "Experiment complete!"
    log "============================================"
}

main "$@"

