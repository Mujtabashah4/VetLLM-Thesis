#!/usr/bin/env python3
"""
Download Llama 3.1 8B Instruct Model
Downloads the model to local directory for faster loading
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, whoami

def load_token_from_file():
    """Load HuggingFace token from common locations"""
    token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
        Path(".hf_token"),
        Path("hf_token.txt"),
    ]
    
    for token_path in token_paths:
        if token_path.exists():
            try:
                token = token_path.read_text().strip()
                if token and token.startswith("hf_"):
                    print(f"✅ Found token at: {token_path}")
                    return token
            except Exception as e:
                continue
    
    # Also check environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        print("✅ Found token in environment variable")
        return token
    
    return None

def check_auth():
    """Check if user is authenticated with HuggingFace"""
    # Try to load token from file first
    token = load_token_from_file()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    
    try:
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info.get('name', 'Unknown')}")
        return True
    except Exception as e:
        error_msg = str(e)
        if "NameResolutionError" in error_msg or "Failed to resolve" in error_msg:
            print(f"⚠️  Network/DNS issue: Cannot resolve huggingface.co")
            print("   But token found - will try to use it directly")
            if token:
                return True  # Continue with token even if whoami fails
        else:
            print(f"❌ Not authenticated: {error_msg[:100]}")
            if not token:
                print("\n📝 Please login to HuggingFace:")
                print("   Run: huggingface-cli login")
                print("   OR: hf auth login")
        return False

def download_model():
    """Download Llama 3.1 8B Instruct model"""
    print("="*80)
    print("📥 Downloading Llama 3.1 8B Instruct Model")
    print("="*80)
    
    # Check authentication
    if not check_auth():
        print("\n⚠️  Please authenticate first, then run this script again.")
        return 1
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_dir = Path("models/llama3.1-8b-instruct")
    
    print(f"\n📋 Model: {model_name}")
    print(f"📁 Download location: {model_dir}")
    print(f"💾 Estimated size: ~16 GB")
    
    # Create directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Checking if model already exists...")
    if (model_dir / "config.json").exists():
        print(f"✅ Model already exists at {model_dir}")
        response = input("   Download again? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipping download.")
            return 0
    
    print("\n📥 Starting download...")
    print("   This may take 10-30 minutes depending on your internet speed.")
    print("   The model will be downloaded with 4-bit quantization support.")
    
    try:
        # Download tokenizer first (fast)
        print("\n   1/2 Downloading tokenizer...")
        # Get token if available
        token = load_token_from_file() or os.environ.get("HF_TOKEN")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(model_dir),
            token=token or True,
        )
        tokenizer.save_pretrained(str(model_dir))
        print("   ✅ Tokenizer downloaded")
        
        # Download model (slow)
        print("\n   2/2 Downloading model (this will take a while)...")
        
        # Use 4-bit quantization to save memory during download
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Set longer timeout for large file downloads
        import os
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 minutes per file
        
        # Get token if available
        token = load_token_from_file() or os.environ.get("HF_TOKEN")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(model_dir),
            token=token or True,
            resume_download=True,  # Resume from partial downloads
        )
        
        # Save model (will save in quantized format)
        print("\n   💾 Saving model...")
        model.save_pretrained(str(model_dir))
        
        print("\n" + "="*80)
        print("✅ Model downloaded successfully!")
        print("="*80)
        print(f"\n📁 Model location: {model_dir}")
        print("\n🚀 You can now start training with:")
        print("   python start_training_llama3.1.py")
        
        return 0
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error downloading model: {error_msg[:200]}...")
        
        # Check if partial download exists
        cache_dir = model_dir / "models--meta-llama--Llama-3.1-8B-Instruct"
        if cache_dir.exists():
            print(f"\n✅ Partial download found at: {cache_dir}")
            print("   The download will automatically resume from where it left off.")
            print("   Just run this script again!")
        
        if "ReadTimeout" in error_msg or "timeout" in error_msg.lower():
            print("\n⏱️  Network timeout detected. This is common for large downloads.")
            print("   The download will automatically resume when you run the script again.")
            print("   HuggingFace caches partial downloads, so you won't lose progress.")
        
        print("\n💡 Troubleshooting:")
        print("   1. ✅ Run the script again - it will resume automatically")
        print("   2. Check your internet connection stability")
        print("   3. Make sure you're logged in: huggingface-cli login")
        print("   4. Ensure you have enough disk space (~20 GB)")
        print("   5. If issues persist, try during off-peak hours")
        
        return 1

if __name__ == "__main__":
    exit_code = download_model()
    sys.exit(exit_code)

