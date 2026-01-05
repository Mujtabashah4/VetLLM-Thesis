#!/usr/bin/env python3
"""
QWEN Fine-tuning Workflow Script
Validates data, shows results, and starts training after confirmation
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from validate_qwen_data import QwenDataValidator

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80)

def check_model_exists():
    """Check if QWEN model exists"""
    model_path = Path("models/qwen2.5-7b-instruct")
    
    if not model_path.exists():
        print(f"❌ ERROR: Model not found at {model_path}")
        print("   Please ensure the QWEN model is downloaded.")
        return False
    
    # Check for model files
    required_files = ["config.json", "tokenizer_config.json"]
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  WARNING: Some model files are missing: {', '.join(missing_files)}")
        print("   Model may not be complete.")
    
    print(f"✅ Model found at: {model_path}")
    return True

def validate_data():
    """Validate the training data"""
    print_header("STEP 1: DATA VALIDATION")
    
    data_dir = Path("experiments/qwen2.5-7b/data")
    train_path = data_dir / "train.json"
    val_path = data_dir / "validation.json"
    test_path = data_dir / "test.json"
    
    # Check if files exist
    if not train_path.exists():
        print(f"❌ ERROR: Training data not found at {train_path}")
        return False, None
    
    if not val_path.exists():
        print(f"❌ ERROR: Validation data not found at {val_path}")
        return False, None
    
    # Run validation
    validator = QwenDataValidator(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path) if test_path.exists() else None
    )
    
    is_valid, stats = validator.validate_all()
    
    return is_valid, stats

def show_data_summary(stats):
    """Show a summary of the data"""
    print_header("DATA SUMMARY")
    
    train_stats = stats.get("train", {})
    val_stats = stats.get("validation", {})
    test_stats = stats.get("test", {})
    
    print("\n📊 Dataset Overview:")
    print(f"  Training samples: {train_stats.get('total_samples', 0)}")
    print(f"  Validation samples: {val_stats.get('total_samples', 0)}")
    print(f"  Test samples: {test_stats.get('total_samples', 0)}")
    
    print("\n📋 Disease Distribution (Training Set):")
    disease_dist = train_stats.get("disease_distribution", {})
    sorted_diseases = sorted(disease_dist.items(), key=lambda x: x[1], reverse=True)
    for disease, count in sorted_diseases[:10]:
        print(f"  {disease}: {count}")
    
    print("\n🐄 Animal Distribution (Training Set):")
    animal_dist = train_stats.get("animal_distribution", {})
    for animal, count in sorted(animal_dist.items()):
        print(f"  {animal}: {count}")
    
    print(f"\n🏷️  SNOMED Code Coverage:")
    print(f"  Training: {train_stats.get('snomed_code_coverage', 0):.1f}%")
    print(f"  Validation: {val_stats.get('snomed_code_coverage', 0):.1f}%")
    
    print(f"\n📏 Text Statistics:")
    print(f"  Avg text length: {train_stats.get('avg_text_length', 0):.1f} chars")
    print(f"  Avg input length: {train_stats.get('avg_input_length', 0):.1f} chars")
    print(f"  Avg output length: {train_stats.get('avg_output_length', 0):.1f} chars")

def get_user_confirmation():
    """Get user confirmation to proceed with training"""
    print_header("TRAINING CONFIRMATION")
    
    print("\n⚠️  READY TO START FINE-TUNING")
    print("\nTraining Configuration:")
    print("  Model: Qwen2.5-7B-Instruct")
    print("  Method: QLoRA (4-bit quantization)")
    print("  Epochs: 3")
    print("  Batch size: 4 per device")
    print("  Gradient accumulation: 4")
    print("  Learning rate: 1e-4")
    print("  Output directory: experiments/qwen2.5-7b/checkpoints")
    
    print("\n" + "-"*80)
    response = input("\nDo you want to proceed with fine-tuning? (yes/no): ").strip().lower()
    
    return response in ['yes', 'y']

def start_training():
    """Start the training process"""
    print_header("STEP 2: STARTING TRAINING")
    
    config_path = Path("experiments/qwen2.5-7b/configs/training_config.yaml")
    train_script = Path("experiments/shared/train.py")
    
    if not config_path.exists():
        print(f"❌ ERROR: Training config not found at {config_path}")
        return False
    
    if not train_script.exists():
        print(f"❌ ERROR: Training script not found at {train_script}")
        return False
    
    print(f"✓ Config file: {config_path}")
    print(f"✓ Training script: {train_script}")
    
    # Create logs directory
    logs_dir = Path("experiments/qwen2.5-7b/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 Starting training...")
    print("   (This may take several hours depending on your hardware)")
    print("   Training logs will be saved to:", logs_dir)
    print("\n" + "-"*80)
    
    # Run training
    try:
        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path)
        ]
        
        # Run training and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print_header("TRAINING COMPLETED SUCCESSFULLY!")
            print("\n✅ Model fine-tuning completed.")
            print(f"   Checkpoints saved to: experiments/qwen2.5-7b/checkpoints")
            return True
        else:
            print_header("TRAINING FAILED")
            print(f"\n❌ Training failed with exit code {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start training: {e}")
        return False

def main():
    """Main workflow"""
    print_header("QWEN FINE-TUNING WORKFLOW")
    
    # Step 1: Check model
    print("\n[Step 0] Checking model...")
    if not check_model_exists():
        print("\n❌ Model check failed. Please ensure the QWEN model is downloaded.")
        return 1
    
    # Step 2: Validate data
    print("\n[Step 1] Validating data...")
    is_valid, stats = validate_data()
    
    if not is_valid:
        print("\n❌ Data validation failed. Please fix the errors before proceeding.")
        return 1
    
    # Step 3: Show summary
    if stats:
        show_data_summary(stats)
    
    # Step 4: Get confirmation
    if not get_user_confirmation():
        print("\n⚠️  Training cancelled by user.")
        return 0
    
    # Step 5: Start training
    success = start_training()
    
    if success:
        print_header("WORKFLOW COMPLETE")
        print("\n✅ Fine-tuning workflow completed successfully!")
        print("\nNext steps:")
        print("  1. Check training logs in: experiments/qwen2.5-7b/logs/")
        print("  2. Evaluate the model using: comprehensive_validation.py")
        print("  3. Use the fine-tuned model for inference")
        return 0
    else:
        print_header("WORKFLOW FAILED")
        print("\n❌ Fine-tuning workflow failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

