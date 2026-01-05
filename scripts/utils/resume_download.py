#!/usr/bin/env python3
"""
Resume Llama 3.1 Download
Quick script to resume interrupted download
"""

import sys
from pathlib import Path

print("="*80)
print("🔄 Resuming Llama 3.1 Download")
print("="*80)

# Check if partial download exists
model_dir = Path("models/llama3.1-8b-instruct")
cache_dir = model_dir / "models--meta-llama--Llama-3.1-8B-Instruct"

if cache_dir.exists():
    import subprocess
    result = subprocess.run(['du', '-sh', str(cache_dir)], capture_output=True, text=True)
    size = result.stdout.split()[0] if result.returncode == 0 else "unknown"
    print(f"\n✅ Found partial download: {size}")
    print("   Resuming download...\n")
else:
    print("\n⚠️  No partial download found. Starting fresh download...\n")

# Run the download script by importing it properly
import importlib.util
spec = importlib.util.spec_from_file_location("download_llama3_1", "download_llama3.1.py")
download_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_module)

exit_code = download_module.download_model()
sys.exit(exit_code)

