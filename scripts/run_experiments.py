#!/usr/bin/env python3
"""
Sample script to run the whole pipeline from CLI
"""

import os

def main():
    print("\nStep 1: Preprocess data")
    os.system("python scripts/data_preprocessing.py")
    print("\nStep 2: Train model")
    os.system("python scripts/train_vetllm.py")
    print("\nStep 3: Evaluate on test set")
    os.system("python scripts/evaluate.py --model-dir models/vetllm-finetuned --data-path data/processed/test_data.json")

if __name__ == "__main__":
    main()
