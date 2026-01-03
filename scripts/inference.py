#!/usr/bin/env python3
"""
VetLLM Inference Script
Generate diagnosis predictions for veterinary clinical notes using fine-tuned model
Supports both base models and LoRA fine-tuned models
"""

import os
import sys
import json
import re
import argparse
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA model loading may fail.")


def create_alpaca_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    """
    Create Alpaca-style prompt format.
    
    Args:
        instruction: The instruction text
        input_text: The input clinical note
        output: The expected output (empty during inference)
    
    Returns:
        Formatted prompt string
    """
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
    return prompt


class VetLLMInference:
    """VetLLM Inference Engine"""
    
    def __init__(
        self,
        model_path: str,
        base_model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        max_new_tokens: int = 100,
        temperature: float = 0.1,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to fine-tuned model (LoRA adapters or full model)
            base_model_name: Base model name if using LoRA (e.g., "wxjiao/alpaca-7b")
            device: Device to use ("cuda", "cpu", "mps", or None for auto)
            max_length: Maximum input sequence length
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        
        # Check if this is a LoRA model
        is_lora = False
        if PEFT_AVAILABLE and os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            is_lora = True
            print("Detected LoRA adapter model")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path if not is_lora else self.base_model_name or self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as e:
            print(f"Warning: Failed to load tokenizer with use_fast=False: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path if not is_lora else self.base_model_name or self.model_path,
                trust_remote_code=True,
            )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if is_lora and PEFT_AVAILABLE:
            if self.base_model_name is None:
                # Try to load base model name from adapter config
                try:
                    config = PeftConfig.from_pretrained(self.model_path)
                    self.base_model_name = config.base_model_name_or_path
                    print(f"Found base model name: {self.base_model_name}")
                except Exception as e:
                    raise ValueError(
                        "LoRA model detected but base_model_name not provided and cannot be inferred. "
                        "Please specify --base-model-name"
                    )
            
            print(f"Loading base model: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            print(f"Loading LoRA adapters from {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
        
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def predict(
        self,
        clinical_note: str,
        instruction: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Predict SNOMED-CT codes for a clinical note.
        
        Args:
            clinical_note: The veterinary clinical note
            instruction: Custom instruction (default: standard instruction)
            max_new_tokens: Override default max_new_tokens
            temperature: Override default temperature
        
        Returns:
            Model prediction string
        """
        if instruction is None:
            instruction = "Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes."
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        if temperature is None:
            temperature = self.temperature
        
        # Create prompt
        prompt = create_alpaca_prompt(
            instruction=instruction,
            input_text=f"Clinical Note: {clinical_note}",
            output=""  # Empty during inference
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def extract_snomed_codes(self, prediction: str) -> List[str]:
        """
        Extract SNOMED-CT codes from model prediction.
        
        Args:
            prediction: Model prediction string
        
        Returns:
            List of extracted SNOMED-CT codes
        """
        # Pattern to match SNOMED codes (typically 6-18 digits)
        codes = re.findall(r'\b\d{6,18}\b', prediction)
        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        return unique_codes


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description="VetLLM Inference - Predict SNOMED-CT codes from clinical notes"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default=None,
        help="Base model name if using LoRA (e.g., 'wxjiao/alpaca-7b')"
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Clinical note text (or use --input-file)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="JSON file with clinical notes (list of strings or list of dicts with 'note' key)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file to save predictions (JSON format)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--extract-codes",
        action="store_true",
        help="Extract and display only SNOMED codes"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = VetLLMInference(
        model_path=args.model,
        base_model_name=args.base_model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Prepare input data
    if args.note:
        notes = [{"note": args.note}]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                notes = data
            else:
                notes = [{"note": note} for note in data]
        else:
            raise ValueError("Input file must contain a list")
    else:
        parser.error("Either --note or --input-file must be provided")
    
    # Run predictions
    results = []
    print("\n" + "="*70)
    print("RUNNING INFERENCE")
    print("="*70)
    
    for i, item in enumerate(notes, 1):
        note = item.get("note", item) if isinstance(item, dict) else item
        
        print(f"\n[{i}/{len(notes)}] Processing clinical note...")
        print(f"Note: {note[:100]}..." if len(note) > 100 else f"Note: {note}")
        
        prediction = inference.predict(note)
        
        if args.extract_codes:
            codes = inference.extract_snomed_codes(prediction)
            print(f"Prediction: {prediction}")
            print(f"Extracted SNOMED codes: {codes}")
            results.append({
                "note": note,
                "prediction": prediction,
                "snomed_codes": codes
            })
        else:
            print(f"Prediction: {prediction}")
            results.append({
                "note": note,
                "prediction": prediction
            })
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to {args.output_file}")
    else:
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
