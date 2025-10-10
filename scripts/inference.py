#!/usr/bin/env python3
"""
VetLLM Inference Script
Generate diagnosis predictions for ad-hoc clinical notes
"""

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def make_prediction(model_path, note, max_new_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    prompt = (
        "Below is an instruction that describes a veterinary diagnosis task, paired with a clinical note. "
        "Write a response that predicts the SNOMED-CT diagnosis codes.\n\n"
        "### Instruction:\nAnalyze this clinical note and provide SNOMED-CT codes.\n\n"
        f"### Input:\nClinical Note: {note}\n\n### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    pred = tokenizer.decode(out_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return pred

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VetLLM inference script")
    parser.add_argument("--model", type=str, required=True, help="Path to model dir")
    parser.add_argument("--note", type=str, required=True, help="Clinical note text")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    output = make_prediction(args.model, args.note, args.max_new_tokens)
    print("\nPrediction:\n", output)

if __name__ == "__main__":
    main()
