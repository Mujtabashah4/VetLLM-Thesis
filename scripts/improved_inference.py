#!/usr/bin/env python3
"""
Improved inference script with post-processing and better prompts.
"""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from post_process_codes import SNOMEDCodeProcessor

class ImprovedVetLLMInference:
    """Improved inference with post-processing."""
    
    def __init__(self, base_model_path, adapter_path):
        """Initialize inference engine."""
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.processor = SNOMEDCodeProcessor()
        
    def load_model(self):
        """Load the fine-tuned model."""
        print("Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        print("✅ Model loaded!")
    
    def create_improved_prompt(self, symptoms: str, animal: str = None) -> str:
        """
        Create improved prompt with explicit formatting instructions.
        Matches training data format exactly.
        """
        if animal:
            input_text = f"Clinical Note: {animal}. Clinical presentation includes {symptoms}. Physical examination reveals these clinical signs."
        else:
            input_text = f"Clinical Note: {symptoms}. Physical examination reveals these clinical signs."
        
        prompt = f"""Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes.

{input_text}

Diagnosed conditions:"""
        
        return prompt
    
    def generate(self, symptoms: str, animal: str = None, max_new_tokens: 128) -> dict:
        """
        Generate diagnosis with post-processing.
        Returns structured result.
        """
        prompt = self.create_improved_prompt(symptoms, animal)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Lower temperature for more consistent output
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetition
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        generated = response[len(prompt_text):].strip()
        
        # Post-process the output
        processed = self.processor.process_model_output(generated)
        
        return {
            'raw_output': generated,
            'prompt': prompt,
            'processed': processed,
            'final_codes': [c['code'] for c in processed['validated_codes']],
            'diseases': processed['diseases'],
            'confidence': processed['confidence']
        }
    
    def diagnose(self, symptoms: str, animal: str = None) -> str:
        """
        Simple diagnosis interface.
        Returns formatted diagnosis string.
        """
        result = self.generate(symptoms, animal)
        
        if result['final_codes']:
            codes = result['final_codes']
            disease_info = ""
            if result['diseases']:
                disease_info = f" ({', '.join(result['diseases'])})"
            return f"Diagnosed conditions: {codes[0]}{disease_info}"
        else:
            return "Diagnosed conditions: [Unable to extract valid SNOMED-CT code]"


def main():
    """Test the improved inference."""
    base_model = "models/alpaca-7b-native"
    adapter_path = "models/vetllm-finetuned"
    
    inference = ImprovedVetLLMInference(base_model, adapter_path)
    inference.load_model()
    
    print("\n" + "=" * 80)
    print("Improved VetLLM Inference Test")
    print("=" * 80)
    
    test_cases = [
        ("high fever, nasal discharge (epistaxis), and sudden death", "Cow"),
        ("high fever, neck swelling, difficulty breathing, and sudden death", "Buffalo"),
        ("swollen udder, drop in milk production, milk in semi-solid form", "Cow"),
    ]
    
    for symptoms, animal in test_cases:
        print(f"\n{'='*80}")
        print(f"Animal: {animal}")
        print(f"Symptoms: {symptoms}")
        print("-" * 80)
        
        result = inference.generate(symptoms, animal)
        
        print(f"Raw Output: {result['raw_output'][:200]}...")
        print(f"Extracted Codes: {result['processed']['extracted_codes']}")
        print(f"Validated Codes: {result['final_codes']}")
        print(f"Diseases: {result['diseases']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nFinal Diagnosis: {inference.diagnose(symptoms, animal)}")


if __name__ == "__main__":
    main()

