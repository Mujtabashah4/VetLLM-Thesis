#!/usr/bin/env python3
"""
Setup basic Python environment for VetLLM.
"""

import os

def main():
    os.system("python -m venv vetllm_env")
    print("Virtual environment created. Activate it via: source vetllm_env/bin/activate")
    os.system("pip install --upgrade pip setuptools wheel")
    pkgs = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "transformers datasets accelerate deepspeed wandb",
        "scikit-learn pandas numpy matplotlib seaborn",
        "peft bitsandbytes"
    ]
    for pkg in pkgs:
        os.system(f"pip install {pkg}")

if __name__ == "__main__":
    main()
