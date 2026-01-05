#!/usr/bin/env python3
"""
Validate the fine-tuned VetLLM model after training.
Tests quality of veterinary diagnosis predictions.
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_path, adapter_path):
    """Load the fine-tuned model."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print("Loading base model with 4-bit quantization...")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

def test_veterinary_cases():
    """Test cases for veterinary diagnosis."""
    return [
        {
            "symptoms": "Cow with high fever, nasal discharge, and difficulty breathing",
            "expected_keywords": ["respiratory", "infection", "pneumonia", "fever"],
        },
        {
            "symptoms": "Buffalo calf with severe diarrhea and dehydration",
            "expected_keywords": ["diarrhea", "dehydration", "calf", "fluid"],
        },
        {
            "symptoms": "Goat showing weakness, loss of appetite, and pale mucous membranes",
            "expected_keywords": ["anemia", "weakness", "parasit", "worm"],
        },
        {
            "symptoms": "Sheep with skin lesions and wool loss",
            "expected_keywords": ["skin", "dermat", "mange", "lesion"],
        },
    ]

def main():
    print("=" * 80)
    print("VetLLM Model Validation")
    print("=" * 80)
    print()
    
    base_model = "models/alpaca-7b-native"
    adapter_path = "models/vetllm-finetuned"
    
    if not Path(adapter_path).exists():
        print(f"❌ Model not found at {adapter_path}")
        print("Please wait for training to complete first.")
        return
    
    print("Loading fine-tuned model...")
    model, tokenizer = load_model(base_model, adapter_path)
    print("✅ Model loaded successfully!")
    print()
    
    test_cases = test_veterinary_cases()
    passed = 0
    total = len(test_cases)
    
    print("Running validation tests...")
    print("-" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total}")
        print(f"Symptoms: {case['symptoms']}")
        
        prompt = f"""You are a veterinary assistant. Analyze the following symptoms and provide a diagnosis.

Symptoms: {case['symptoms']}

Diagnosis:"""
        
        response = generate_response(model, tokenizer, prompt)
        print(f"Response: {response[:300]}...")
        
        # Check for expected keywords
        response_lower = response.lower()
        found_keywords = [kw for kw in case['expected_keywords'] if kw.lower() in response_lower]
        
        if len(found_keywords) >= len(case['expected_keywords']) // 2:
            print(f"✅ PASSED (Found keywords: {found_keywords})")
            passed += 1
        else:
            print(f"⚠️  PARTIAL (Found: {found_keywords}, Expected some of: {case['expected_keywords']})")
    
    print()
    print("=" * 80)
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed >= total * 0.75:
        print("✅ Model quality is GOOD - ready for use!")
    else:
        print("⚠️  Model may need more training or data refinement")

if __name__ == "__main__":
    main()

