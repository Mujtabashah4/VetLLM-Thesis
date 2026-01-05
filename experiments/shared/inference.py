"""
VetLLM Inference Script
Interactive inference for veterinary diagnosis

Supports:
- Base model inference (zero-shot)
- Fine-tuned model inference
- Batch processing
- Interactive CLI mode
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel


# ============================================
# Chat Format Templates
# ============================================

class ChatFormatter:
    """Format prompts for different model architectures."""
    
    SYSTEM_PROMPT = """You are VetLLM, a veterinary clinical assistant specialized in livestock diseases. 
Given a clinical case presentation, you will:
1. Analyze the clinical signs and symptoms
2. Provide the most likely diagnosis with SNOMED-CT codes when available
3. List differential diagnoses to consider
4. Recommend appropriate treatment and management steps
5. Explain your clinical reasoning"""

    @staticmethod
    def format_llama31(species: str, symptoms: str) -> str:
        """Format prompt for Llama 3.1 8B Instruct."""
        user_content = f"""Analyze this veterinary case:

Species: {species}
Clinical presentation: {symptoms}

Please provide:
1. Most likely diagnosis (with SNOMED-CT code if available)
2. Differential diagnoses
3. Recommended treatment plan
4. Clinical reasoning"""
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{ChatFormatter.SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def format_qwen25(species: str, symptoms: str) -> str:
        """Format prompt for Qwen2.5 7B Instruct (ChatML)."""
        user_content = f"""Analyze this veterinary case:

**Species**: {species}
**Clinical Signs**: {symptoms}

Provide your diagnosis, differentials, treatment plan, and reasoning."""
        
        return f"""<|im_start|>system
{ChatFormatter.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


# ============================================
# Model Loader
# ============================================

@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name_or_path: str
    adapter_path: Optional[str] = None  # Path to LoRA adapter
    use_4bit: bool = True
    use_flash_attention: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"


def load_model(config: ModelConfig):
    """Load model and tokenizer."""
    print(f"Loading model: {config.model_name_or_path}")
    
    # Quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Torch dtype
    torch_dtype = getattr(torch, config.torch_dtype)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    attn_impl = "flash_attention_2" if config.use_flash_attention and torch.cuda.is_available() else None
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    
    # Load adapter if specified
    if config.adapter_path:
        print(f"Loading adapter from: {config.adapter_path}")
        model = PeftModel.from_pretrained(model, config.adapter_path)
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


# ============================================
# Inference Functions
# ============================================

def generate_diagnosis(
    model,
    tokenizer,
    species: str,
    symptoms: str,
    model_type: str = "llama3.1",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate a diagnosis for a veterinary case.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        species: Animal species (e.g., "Cow", "Goat")
        symptoms: Clinical symptoms description
        model_type: "llama3.1" or "qwen2.5"
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (False for greedy)
    
    Returns:
        Generated diagnosis response
    """
    # Format prompt based on model type
    if model_type == "llama3.1":
        prompt = ChatFormatter.format_llama31(species, symptoms)
    elif model_type == "qwen2.5":
        prompt = ChatFormatter.format_qwen25(species, symptoms)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    response = full_response[len(prompt_decoded):].strip()
    
    return response


def batch_inference(
    model,
    tokenizer,
    cases: List[Dict[str, str]],
    model_type: str = "llama3.1",
    **generation_kwargs
) -> List[Dict[str, Any]]:
    """
    Run inference on a batch of cases.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        cases: List of dicts with 'species' and 'symptoms' keys
        model_type: "llama3.1" or "qwen2.5"
        **generation_kwargs: Additional generation parameters
    
    Returns:
        List of results with input and output
    """
    results = []
    
    for i, case in enumerate(cases):
        print(f"Processing case {i+1}/{len(cases)}...")
        
        response = generate_diagnosis(
            model,
            tokenizer,
            species=case['species'],
            symptoms=case['symptoms'],
            model_type=model_type,
            **generation_kwargs
        )
        
        results.append({
            'input': case,
            'output': response,
        })
    
    return results


# ============================================
# Interactive CLI
# ============================================

def interactive_mode(
    model,
    tokenizer,
    model_type: str = "llama3.1"
):
    """Run interactive inference mode."""
    print("\n" + "=" * 60)
    print("VetLLM Interactive Diagnosis Mode")
    print("=" * 60)
    print("\nEnter case details to get a diagnosis.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get species
        species = input("Species (e.g., Cow, Goat, Sheep, Buffalo): ").strip()
        if species.lower() in ['quit', 'exit']:
            break
        
        if not species:
            print("Please enter a species.")
            continue
        
        # Get symptoms
        print("Enter symptoms (press Enter twice to finish):")
        symptoms_lines = []
        while True:
            line = input()
            if line == "":
                break
            symptoms_lines.append(line)
        
        symptoms = " ".join(symptoms_lines)
        if not symptoms:
            print("Please enter symptoms.")
            continue
        
        # Generate diagnosis
        print("\n" + "-" * 40)
        print("Generating diagnosis...")
        print("-" * 40 + "\n")
        
        response = generate_diagnosis(
            model,
            tokenizer,
            species=species,
            symptoms=symptoms,
            model_type=model_type,
        )
        
        print(response)
        print("\n" + "=" * 60 + "\n")


# ============================================
# Comparison Mode
# ============================================

def compare_models(
    base_model_config: ModelConfig,
    finetuned_model_config: ModelConfig,
    test_cases: List[Dict[str, str]],
    model_type: str = "llama3.1",
    output_path: Optional[str] = None,
):
    """
    Compare base model vs fine-tuned model on test cases.
    
    Args:
        base_model_config: Config for base model
        finetuned_model_config: Config for fine-tuned model
        test_cases: List of test cases
        model_type: "llama3.1" or "qwen2.5"
        output_path: Optional path to save comparison results
    """
    results = []
    
    # Load base model
    print("\n" + "=" * 60)
    print("Loading Base Model (Zero-shot)")
    print("=" * 60)
    base_model, base_tokenizer = load_model(base_model_config)
    
    # Run base model inference
    print("\nRunning base model inference...")
    base_results = batch_inference(
        base_model, base_tokenizer, test_cases, model_type,
        do_sample=False, temperature=0.1
    )
    
    # Clear base model from memory
    del base_model
    torch.cuda.empty_cache()
    
    # Load fine-tuned model
    print("\n" + "=" * 60)
    print("Loading Fine-tuned Model")
    print("=" * 60)
    ft_model, ft_tokenizer = load_model(finetuned_model_config)
    
    # Run fine-tuned model inference
    print("\nRunning fine-tuned model inference...")
    ft_results = batch_inference(
        ft_model, ft_tokenizer, test_cases, model_type,
        do_sample=False, temperature=0.1
    )
    
    # Combine results
    for i, (base_res, ft_res) in enumerate(zip(base_results, ft_results)):
        results.append({
            'case_id': i + 1,
            'input': base_res['input'],
            'base_model_output': base_res['output'],
            'finetuned_model_output': ft_res['output'],
        })
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nComparison results saved to: {output_path}")
    
    # Print sample comparison
    print("\n" + "=" * 60)
    print("Sample Comparison (Case 1)")
    print("=" * 60)
    
    if results:
        sample = results[0]
        print(f"\nInput: {sample['input']}")
        print("\n--- Base Model (Zero-shot) ---")
        print(sample['base_model_output'][:500] + "..." if len(sample['base_model_output']) > 500 else sample['base_model_output'])
        print("\n--- Fine-tuned Model ---")
        print(sample['finetuned_model_output'][:500] + "..." if len(sample['finetuned_model_output']) > 500 else sample['finetuned_model_output'])
    
    return results


# ============================================
# CLI Interface
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VetLLM Inference Script")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive diagnosis mode")
    interactive_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    interactive_parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    interactive_parser.add_argument("--model-type", type=str, choices=["llama3.1", "qwen2.5"], default="llama3.1")
    interactive_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    # Single inference
    single_parser = subparsers.add_parser("single", help="Single case inference")
    single_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    single_parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    single_parser.add_argument("--model-type", type=str, choices=["llama3.1", "qwen2.5"], default="llama3.1")
    single_parser.add_argument("--species", type=str, required=True, help="Animal species")
    single_parser.add_argument("--symptoms", type=str, required=True, help="Clinical symptoms")
    single_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    # Batch inference
    batch_parser = subparsers.add_parser("batch", help="Batch inference on JSON file")
    batch_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    batch_parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    batch_parser.add_argument("--model-type", type=str, choices=["llama3.1", "qwen2.5"], default="llama3.1")
    batch_parser.add_argument("--input", type=str, required=True, help="Input JSON file with cases")
    batch_parser.add_argument("--output", type=str, required=True, help="Output JSON file for results")
    batch_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    if args.command == "interactive":
        config = ModelConfig(
            model_name_or_path=args.model,
            adapter_path=args.adapter,
            use_4bit=not args.no_4bit,
        )
        model, tokenizer = load_model(config)
        interactive_mode(model, tokenizer, args.model_type)
    
    elif args.command == "single":
        config = ModelConfig(
            model_name_or_path=args.model,
            adapter_path=args.adapter,
            use_4bit=not args.no_4bit,
        )
        model, tokenizer = load_model(config)
        
        response = generate_diagnosis(
            model, tokenizer,
            species=args.species,
            symptoms=args.symptoms,
            model_type=args.model_type,
        )
        
        print("\n" + "=" * 60)
        print("Diagnosis Result")
        print("=" * 60)
        print(response)
    
    elif args.command == "batch":
        config = ModelConfig(
            model_name_or_path=args.model,
            adapter_path=args.adapter,
            use_4bit=not args.no_4bit,
        )
        model, tokenizer = load_model(config)
        
        # Load input cases
        with open(args.input, 'r') as f:
            cases = json.load(f)
        
        # Run batch inference
        results = batch_inference(
            model, tokenizer, cases,
            model_type=args.model_type,
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    else:
        parser.print_help()

