#!/usr/bin/env python3
"""
VetLLM Evaluation Script
Evaluates fine-tuned VetLLM model on test or validation datasets
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def extract_snomed_codes(prediction: str) -> List[str]:
    import re
    codes = []
    code_patterns = [
        r'\b\d{6,18}\b',
        r'SNOMED[-_]?CT?\s*:?\s*(\d{6,18})',
        r'Code\s*:?\s*(\d{6,18})',
    ]
    for pattern in code_patterns:
        matches = re.findall(pattern, prediction, re.IGNORECASE)
        codes.extend(matches)
    if not codes:
        terms = re.split(r'[,;.\n]', prediction)
        for term in terms:
            t = term.strip()
            if t and len(t) > 3:
                codes.append(t)
    return list(dict.fromkeys(codes))[:10]

def evaluate_file(model_dir, data_path, max_new_tokens=150):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with open(data_path, 'r') as f:
        samples = json.load(f)

    all_y_true = []
    all_y_pred = []

    results = []
    for item in samples:
        prompt = (
            "Below is an instruction that describes a veterinary diagnosis task, paired with a clinical note. "
            "Write a response that predicts the SNOMED-CT diagnosis codes.\n\n"
            f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        pred = tokenizer.decode(out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_codes = extract_snomed_codes(pred)
        true_codes = item.get("snomed_codes", [])
        all_y_pred.append(set(pred_codes))
        all_y_true.append(set(true_codes))
        results.append({"input": item['input'], "truth": true_codes, "pred": pred_codes, "prompt": prompt})

    # Calculate metrics
    exact_matches = [int(y_p == y_t) for y_p, y_t in zip(all_y_pred, all_y_true)]
    jaccards    = [
        len(y_p.intersection(y_t)) / len(y_p.union(y_t)) if y_p.union(y_t) else 1.0
        for y_p, y_t in zip(all_y_pred, all_y_true)
    ]
    avg_exact = np.mean(exact_matches)
    avg_jaccard = np.mean(jaccards)

    result_stats = {
        "samples": len(samples),
        "exact_match": float(avg_exact),
        "avg_jaccard": float(avg_jaccard)
    }
    print(json.dumps(result_stats, indent=2))
    return results, result_stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate VetLLM model")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained model dir")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data file (test/val)")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--output", type=str, help="Optional: path to write detailed results as json")
    args = parser.parse_args()

    results, stats = evaluate_file(args.model_dir, args.data_path, args.max_new_tokens)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"results": results, "stats": stats}, f, indent=2)
        print(f"Wrote detailed results to {args.output}")

if __name__ == "__main__":
    main()
