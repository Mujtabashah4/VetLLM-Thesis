#!/usr/bin/env python3
"""
Quick inference test for QWEN fine-tuned model
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "experiments" / "shared"))
from inference import ModelConfig, load_model, generate_diagnosis

def main():
    """Test inference on sample cases."""
    print("="*80)
    print("QWEN MODEL INFERENCE TEST")
    print("="*80)
    
    # Model configuration
    config = ModelConfig(
        model_name_or_path="/home/iml_admin/Desktop/VetLLM/VetLLM-Thesis/models/qwen2.5-7b-instruct",
        adapter_path="/home/iml_admin/Desktop/VetLLM/VetLLM-Thesis/experiments/qwen2.5-7b/checkpoints/final",
        use_4bit=True,  # Use quantization for inference
        use_flash_attention=False,  # Disable flash attention
    )
    
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model(config)
    print("✅ Model loaded successfully!")
    
    # Test cases
    test_cases = [
        {
            "species": "Sheep",
            "symptoms": "fever, labial vesicles, nasal discharge, and bloody diarrhea",
            "expected": "Peste des Petits Ruminants (SNOMED-CT: 1679004)"
        },
        {
            "species": "Cow",
            "symptoms": "high fever, persistent diarrhea with blood, and dehydration",
            "expected": "Anthrax (SNOMED-CT: 40214000)"
        },
        {
            "species": "Buffalo",
            "symptoms": "high fever, neck swelling, difficulty breathing",
            "expected": "Hemorrhagic Septicemia (SNOMED-CT: 198462004)"
        },
        {
            "species": "Cow",
            "symptoms": "swollen udder, drop in milk production, blood in milk",
            "expected": "Mastitis (SNOMED-CT: 72934000)"
        },
        {
            "species": "Goat",
            "symptoms": "severe cough, difficulty breathing, rapid breathing, fever",
            "expected": "Contagious Caprine Pleuropneumonia (SNOMED-CT: 2260006)"
        }
    ]
    
    print("\n[2/3] Running inference on test cases...")
    print("="*80)
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}/{len(test_cases)}]")
        print(f"Species: {case['species']}")
        print(f"Symptoms: {case['symptoms']}")
        print(f"Expected: {case['expected']}")
        print("-" * 80)
        
        try:
            response = generate_diagnosis(
                model, tokenizer,
                species=case['species'],
                symptoms=case['symptoms'],
                model_type="qwen2.5",
            )
            
            print(f"Model Response:\n{response}")
            
            results.append({
                "test_case": i,
                "species": case['species'],
                "symptoms": case['symptoms'],
                "expected": case['expected'],
                "prediction": response
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "test_case": i,
                "species": case['species'],
                "symptoms": case['symptoms'],
                "expected": case['expected'],
                "error": str(e)
            })
    
    print("\n[3/3] Saving results...")
    output_file = "reports/qwen_inference_test.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    print("\n" + "="*80)
    print("INFERENCE TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

