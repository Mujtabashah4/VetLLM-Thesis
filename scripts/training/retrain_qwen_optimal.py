#!/usr/bin/env python3
"""
Retrain QWEN with optimal configuration
- 7 epochs maximum
- Early stopping to prevent overfitting
- Best model automatically saved based on validation loss
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Retrain QWEN with optimal settings."""
    print("="*80)
    print("QWEN OPTIMAL RETRAINING")
    print("="*80)
    print("\nConfiguration:")
    print("  - Epochs: 7 (with early stopping)")
    print("  - Evaluation: Every epoch (24 steps)")
    print("  - Early Stopping: Patience=3, Threshold=0.001")
    print("  - Best Model: Automatically saved based on validation loss")
    print("\nThis will:")
    print("  ✅ Train for up to 7 epochs")
    print("  ✅ Stop early if validation loss doesn't improve")
    print("  ✅ Save the best model (lowest validation loss)")
    print("  ✅ Overwrite checkpoints/final/ with best model")
    print("\n" + "="*80)
    
    response = input("\nProceed with optimal retraining? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    config_path = "experiments/qwen2.5-7b/configs/training_config.yaml"
    train_script = "experiments/shared/train.py"
    
    print("\n🚀 Starting optimal training...")
    print("   (This will take ~10-12 minutes)")
    print("   Monitor progress: tail -f training_optimal.log")
    print("\n" + "-"*80)
    
    # Run training
    cmd = [
        sys.executable,
        train_script,
        "--config",
        config_path
    ]
    
    with open("training_optimal.log", "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()
        
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "="*80)
            print("✅ OPTIMAL TRAINING COMPLETED!")
            print("="*80)
            print("\nBest model saved to: experiments/qwen2.5-7b/checkpoints/final/")
            print("This model has the lowest validation loss and is ready for:")
            print("  - Testing")
            print("  - Validation")
            print("  - Inference")
            return 0
        else:
            print("\n" + "="*80)
            print("❌ TRAINING FAILED")
            print("="*80)
            return 1

if __name__ == "__main__":
    sys.exit(main())

