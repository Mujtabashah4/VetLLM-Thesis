#!/usr/bin/env python3
"""
Download Alpaca-7B Model Script
Downloads the base Alpaca model from Hugging Face Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

def check_disk_space(required_gb=15):
    """Check if sufficient disk space is available"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    
    if free_gb < required_gb:
        print(f"⚠️  Warning: Only {free_gb}GB free space available. Recommended: {required_gb}GB")
        return False
    return True

def download_alpaca_model(model_name="wxjiao/alpaca-7b", local_dir="./"):
    """Download Alpaca model and tokenizer"""
    
    print(f"🚀 Starting download of {model_name}")
    print("="*50)
    
    # Check disk space
    if not check_disk_space():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Download model files
        print("📥 Downloading model files...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("✅ Model files downloaded successfully")
        
        # Verify model can be loaded
        print("🔍 Verifying model integrity...")
        
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print("✅ Model verification successful")
        
        # Save download info
        download_info = {
            "model_name": model_name,
            "download_date": str(Path().cwd()),
            "local_path": str(Path(local_dir).absolute()),
            "model_size": f"{sum(f.stat().st_size for f in Path(local_dir).rglob('*') if f.is_file()) / (1024**3):.2f} GB",
            "files_downloaded": [f.name for f in Path(local_dir).iterdir() if f.is_file()],
            "verification_passed": True
        }
        
        with open(Path(local_dir) / "download_info.json", 'w') as f:
            json.dump(download_info, f, indent=2)
        
        print(f"📁 Model saved to: {Path(local_dir).absolute()}")
        print("🎉 Download completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

def main():
    """Main download function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Alpaca-7B model")
    parser.add_argument("--model", default="wxjiao/alpaca-7b", help="Model name on Hugging Face")
    parser.add_argument("--output", default="./", help="Output directory")
    parser.add_argument("--token", help="Hugging Face token (if required)")
    
    args = parser.parse_args()
    
    # Login if token provided
    if args.token:
        login(token=args.token)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download model
    success = download_alpaca_model(args.model, str(output_path))
    
    if success:
        print("\n" + "="*50)
        print("🎯 Next steps:")
        print("1. Verify model files in the output directory")
        print("2. Run the VetLLM training pipeline")
        print("3. Check download_info.json for details")
    else:
        print("\n❌ Download failed. Check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
