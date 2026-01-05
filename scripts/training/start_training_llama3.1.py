#!/usr/bin/env python3
"""
Start Llama 3.1 8B Fine-tuning
Downloads model if needed and starts training
"""

import os
import sys
import subprocess
from pathlib import Path
import torch

def check_gpu():
    """Check GPU availability"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Training requires GPU.")
        return False
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU Available: {gpu_name}")
    print(f"   GPU Count: {gpu_count}")
    print(f"   GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 20:
        print("⚠️  Warning: GPU memory is less than 20GB. Using 4-bit quantization (QLoRA).")
    
    return True

def check_model_access():
    """Check if we can access the Llama 3.1 model"""
    print("\n🔍 Checking Llama 3.1 model access...")
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    local_model_dir = Path("models/llama3.1-8b-instruct")
    
    # Check if model is already downloaded locally
    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        print(f"   ✅ Found local model at: {local_model_dir}")
        print("   Using local model (faster startup)")
        return True
    
    # If not local, check if we can access from HuggingFace
    print(f"   ⚠️  Local model not found. Will download from HuggingFace during training.")
    print(f"   💡 Tip: Download first with 'python download_llama3.1.py' for faster startup")
    
    try:
        from transformers import AutoTokenizer
        
        print(f"   Checking access to: {model_name}")
        
        # Try to load tokenizer (lightweight check)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=True,  # Will use HuggingFace token if set
        )
        
        print("   ✅ Model access confirmed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error accessing model: {e}")
        print("\n📝 To access Llama 3.1, you need to:")
        print("   1. Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("   2. Accept the license")
        print("   3. Login to HuggingFace:")
        print("      huggingface-cli login")
        print("\n💡 Or download the model first:")
        print("      python download_llama3.1.py")
        return False

def check_data():
    """Check if data files exist"""
    print("\n🔍 Checking data files...")
    
    data_dir = Path("experiments/llama3.1-8b/data")
    required_files = ["train.json", "validation.json", "test.json"]
    
    all_exist = True
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("✅ All data files found!")
        return True
    else:
        print("❌ Missing data files. Please prepare data first.")
        return False

def start_training():
    """Start the training process"""
    print("\n" + "="*80)
    print("🚀 Starting Llama 3.1 8B Fine-tuning")
    print("="*80)
    
    # Check prerequisites
    if not check_gpu():
        return 1
    
    if not check_model_access():
        return 1
    
    if not check_data():
        return 1
    
    # Configuration
    config_path = Path("experiments/llama3.1-8b/configs/training_config.yaml")
    train_script = Path("experiments/shared/train.py")
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1
    
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        return 1
    
    print("\n📋 Training Configuration:")
    print("   Model: meta-llama/Llama-3.1-8B-Instruct")
    print("   Method: QLoRA (4-bit quantization)")
    print("   Epochs: 3")
    print("   Batch Size: 4 per device × 4 gradient accumulation = 16 effective")
    print("   Learning Rate: 1e-4")
    print("   Optimizer: paged_adamw_8bit")
    print("   Output: experiments/llama3.1-8b/checkpoints/")
    
    print("\n" + "="*80)
    print("Starting training in 5 seconds...")
    print("Press Ctrl+C to cancel")
    print("="*80)
    
    import time
    for i in range(5, 0, -1):
        print(f"   {i}...", end='\r')
        time.sleep(1)
    
    print("\n🚀 Starting training now!\n")
    
    # Set environment variables
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    # Run training
    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path)
    ]
    
    try:
        # Run training and stream output
        process = subprocess.Popen(
            cmd,
            env=env,
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
            print("\n" + "="*80)
            print("✅ Training completed successfully!")
            print("="*80)
            print(f"\n📁 Model saved to: experiments/llama3.1-8b/checkpoints/final/")
            print(f"📊 Logs saved to: experiments/llama3.1-8b/logs/")
            return 0
        else:
            print("\n" + "="*80)
            print("❌ Training failed with exit code:", process.returncode)
            print("="*80)
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return 1

if __name__ == "__main__":
    exit_code = start_training()
    sys.exit(exit_code)

