#!/usr/bin/env python3
"""
Validation script for QWEN fine-tuned model
Adapted from comprehensive_validation.py for QWEN model
"""

import sys
from pathlib import Path

# Import the validation class
sys.path.insert(0, str(Path(__file__).parent))
from comprehensive_validation import VetLLMValidator

def main():
    """Main validation function for QWEN model."""
    base_model = "/home/iml_admin/Desktop/VetLLM/VetLLM-Thesis/models/qwen2.5-7b-instruct"
    adapter_path = "/home/iml_admin/Desktop/VetLLM/VetLLM-Thesis/experiments/qwen2.5-7b/checkpoints/final"
    
    print("="*80)
    print("QWEN MODEL VALIDATION")
    print("="*80)
    print(f"Base Model: {base_model}")
    print(f"Adapter Path: {adapter_path}")
    print("="*80)
    
    validator = VetLLMValidator(base_model, adapter_path, use_post_processing=True)
    validator.load_model()
    validator.run_validation()
    validator.print_results()
    validator.save_results('reports/qwen_validation_results.json')
    
    print("\n✅ QWEN model validation complete!")
    print(f"Results saved to: reports/qwen_validation_results.json")

if __name__ == "__main__":
    main()

