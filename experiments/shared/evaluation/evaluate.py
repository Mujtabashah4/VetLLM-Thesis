"""
VetLLM Evaluation Pipeline
Comprehensive evaluation for veterinary diagnosis models

Metrics:
- Classification: Accuracy, F1 (macro/micro/weighted), Precision, Recall
- Generation: BLEU, ROUGE, BERTScore
- Clinical: Diagnosis accuracy, differential coverage, treatment relevance
"""

import os
import sys
import json
import re
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# Optional imports for generation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ============================================
# Configuration
# ============================================

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    model_path: str  # Path to fine-tuned model or base model
    test_data_path: str  # Path to test.json
    output_dir: str  # Where to save results
    
    # Model settings
    base_model_name: Optional[str] = None  # For adapter models
    is_adapter: bool = True  # Whether model_path is a LoRA adapter
    use_4bit: bool = True  # Use 4-bit quantization
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False  # Greedy for evaluation
    
    # Evaluation settings
    batch_size: int = 1  # Generation batch size
    compute_generation_metrics: bool = True
    num_samples: Optional[int] = None  # Limit samples for quick eval


# ============================================
# Diagnosis Extraction
# ============================================

def extract_diagnosis_from_output(text: str) -> Tuple[str, List[str], str]:
    """
    Extract diagnosis, differentials, and treatment from model output.
    
    Returns:
        Tuple of (primary_diagnosis, differential_list, treatment)
    """
    primary_diagnosis = ""
    differentials = []
    treatment = ""
    
    # Extract primary diagnosis
    # Pattern: "Primary Diagnosis: **Disease** (SNOMED-CT: ...)"
    diag_patterns = [
        r'\*\*Primary Diagnosis\*\*:\s*\*\*([^*]+)\*\*',
        r'Primary Diagnosis:\s*\*\*([^*]+)\*\*',
        r'1\.\s*\*\*Primary Diagnosis\*\*:\s*\*\*([^*]+)\*\*',
        r'1\.\s*Diagnosis:\s*([^\n]+)',
        r'Diagnosed conditions:\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in diag_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            primary_diagnosis = match.group(1).strip()
            # Remove SNOMED codes from diagnosis name
            primary_diagnosis = re.sub(r'\s*\(SNOMED-CT:.*?\)', '', primary_diagnosis)
            break
    
    # Extract differentials
    diff_patterns = [
        r'\*\*Differential Diagnoses\*\*:\s*((?:\s*-\s*[^\n]+\n?)+)',
        r'Differential Diagnoses:\s*((?:\s*-\s*[^\n]+\n?)+)',
        r'2\.\s*\*\*Differential Diagnoses\*\*:\s*((?:\s*-\s*[^\n]+\n?)+)',
    ]
    
    for pattern in diff_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            diff_text = match.group(1)
            differentials = re.findall(r'-\s*([^\n]+)', diff_text)
            differentials = [d.strip() for d in differentials if d.strip()]
            break
    
    # Extract treatment
    treat_patterns = [
        r'\*\*Recommended Treatment\*\*:\s*([^\n]+(?:\n(?!\d\.)[^\n]+)*)',
        r'3\.\s*\*\*Recommended Treatment\*\*:\s*([^\n]+)',
        r'Treatment:\s*([^\n]+)',
    ]
    
    for pattern in treat_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            treatment = match.group(1).strip()
            break
    
    return primary_diagnosis, differentials, treatment


def normalize_diagnosis_name(diagnosis: str) -> str:
    """Normalize diagnosis name for comparison."""
    # Remove special characters and normalize
    normalized = diagnosis.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Handle common abbreviations
    abbreviations = {
        'fmd': 'foot and mouth disease',
        'ppr': 'peste des petits ruminants',
        'bq': 'black quarter',
        'hs': 'hemorrhagic septicemia',
        'tb': 'tuberculosis',
        'ccpp': 'contagious caprine pleuropneumonia',
    }
    
    if normalized in abbreviations:
        normalized = abbreviations[normalized]
    
    return normalized


# ============================================
# Metrics Computation
# ============================================

class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if BLEU_AVAILABLE else None
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        ) if ROUGE_AVAILABLE else None
    
    def compute_classification_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute classification metrics for diagnosis prediction."""
        # Normalize names
        pred_normalized = [normalize_diagnosis_name(p) for p in predictions]
        ref_normalized = [normalize_diagnosis_name(r) for r in references]
        
        # Get unique labels
        all_labels = list(set(ref_normalized))
        
        # Create label mapping
        label_to_id = {label: i for i, label in enumerate(all_labels)}
        
        # Convert to numeric (handle unknown predictions)
        y_true = [label_to_id.get(r, -1) for r in ref_normalized]
        y_pred = []
        for p in pred_normalized:
            if p in label_to_id:
                y_pred.append(label_to_id[p])
            else:
                # Find closest match or assign unknown
                y_pred.append(-1)
        
        # Filter out unknowns for sklearn metrics
        valid_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != -1]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        # Handle empty predictions
        if not y_true_valid:
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_micro': 0.0,
                'f1_weighted': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
            }
        
        # Exact match accuracy (original strings)
        exact_matches = sum(1 for p, r in zip(pred_normalized, ref_normalized) if p == r)
        accuracy = exact_matches / len(predictions) if predictions else 0.0
        
        # Sklearn metrics (on valid predictions)
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true_valid, y_pred_valid, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
        }
        
        return metrics
    
    def compute_bleu(self, prediction: str, reference: str) -> float:
        """Compute BLEU score between prediction and reference."""
        if not BLEU_AVAILABLE:
            return 0.0
        
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        try:
            return sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=self.smoothing
            )
        except Exception:
            return 0.0
    
    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores between prediction and reference."""
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
            }
        except Exception:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def compute_generation_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute generation metrics (BLEU, ROUGE) over all samples."""
        bleu_scores = []
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            bleu_scores.append(self.compute_bleu(pred, ref))
            
            rouge = self.compute_rouge(pred, ref)
            for key, value in rouge.items():
                rouge_scores[key].append(value)
        
        metrics = {
            'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        }
        
        for key, values in rouge_scores.items():
            metrics[key] = np.mean(values) if values else 0.0
        
        return metrics


# ============================================
# Model Loading
# ============================================

def load_model_for_evaluation(config: EvaluationConfig):
    """Load model and tokenizer for evaluation."""
    
    # Quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    if config.is_adapter:
        # Load base model first
        base_model_name = config.base_model_name or config.model_path
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left",  # For generation
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter
        if config.model_path != base_model_name:
            model = PeftModel.from_pretrained(model, config.model_path)
            model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Load full model
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


# ============================================
# Evaluation Pipeline
# ============================================

def generate_predictions(
    model,
    tokenizer,
    test_data: List[Dict],
    config: EvaluationConfig,
) -> List[Dict]:
    """Generate predictions for test data."""
    results = []
    
    for sample in tqdm(test_data, desc="Generating predictions"):
        # Get input (the user query part)
        input_text = sample.get('input', '')
        
        # Reconstruct the prompt based on model type
        # Detect format from the 'text' field
        full_text = sample.get('text', '')
        
        if '<|begin_of_text|>' in full_text:
            # Llama format - extract up to assistant header
            # Find where assistant response starts
            assistant_marker = '<|start_header_id|>assistant<|end_header_id|>'
            if assistant_marker in full_text:
                prompt = full_text.split(assistant_marker)[0] + assistant_marker + '\n\n'
            else:
                prompt = full_text
        elif '<|im_start|>' in full_text:
            # Qwen format
            assistant_marker = '<|im_start|>assistant'
            if assistant_marker in full_text:
                prompt = full_text.split(assistant_marker)[0] + assistant_marker + '\n'
            else:
                prompt = full_text
        else:
            # Fallback - use the input directly
            prompt = input_text
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_new_tokens,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (after the prompt)
        prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        generated_response = generated_text[len(prompt_decoded):].strip()
        
        # Extract components from generated response
        pred_diagnosis, pred_differentials, pred_treatment = extract_diagnosis_from_output(generated_response)
        
        # Get reference values
        metadata = sample.get('metadata', {})
        ref_diagnosis = metadata.get('disease_normalized', metadata.get('disease', ''))
        
        results.append({
            'input': input_text,
            'reference_output': sample.get('output', ''),
            'generated_output': generated_response,
            'reference_diagnosis': ref_diagnosis,
            'predicted_diagnosis': pred_diagnosis,
            'predicted_differentials': pred_differentials,
            'predicted_treatment': pred_treatment,
            'metadata': metadata,
        })
    
    return results


def evaluate(config: EvaluationConfig) -> Dict[str, Any]:
    """Run full evaluation pipeline."""
    
    print("=" * 60)
    print("VetLLM Evaluation Pipeline")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"\nLoading test data from: {config.test_data_path}")
    with open(config.test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Limit samples if specified
    if config.num_samples:
        test_data = test_data[:config.num_samples]
    
    print(f"Test samples: {len(test_data)}")
    
    # Load model
    print(f"\nLoading model from: {config.model_path}")
    model, tokenizer = load_model_for_evaluation(config)
    
    # Generate predictions
    print("\nGenerating predictions...")
    results = generate_predictions(model, tokenizer, test_data, config)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    calculator = MetricsCalculator()
    
    # Extract predictions and references
    predictions = [r['predicted_diagnosis'] for r in results]
    references = [r['reference_diagnosis'] for r in results]
    
    # Classification metrics
    classification_metrics = calculator.compute_classification_metrics(predictions, references)
    
    # Generation metrics (on full outputs)
    generation_metrics = {}
    if config.compute_generation_metrics:
        pred_outputs = [r['generated_output'] for r in results]
        ref_outputs = [r['reference_output'] for r in results]
        generation_metrics = calculator.compute_generation_metrics(pred_outputs, ref_outputs)
    
    # Aggregate results
    evaluation_results = {
        'config': {
            'model_path': config.model_path,
            'test_data_path': config.test_data_path,
            'num_samples': len(test_data),
        },
        'classification_metrics': classification_metrics,
        'generation_metrics': generation_metrics,
        'predictions': results,
    }
    
    # Save results
    results_path = output_path / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save summary
    summary = {
        'model': config.model_path,
        'num_samples': len(test_data),
        'accuracy': classification_metrics['accuracy'],
        'f1_macro': classification_metrics['f1_macro'],
        'f1_weighted': classification_metrics['f1_weighted'],
        'precision_macro': classification_metrics['precision_macro'],
        'recall_macro': classification_metrics['recall_macro'],
    }
    if generation_metrics:
        summary.update({
            'bleu': generation_metrics.get('bleu', 0.0),
            'rouge1': generation_metrics.get('rouge1', 0.0),
            'rougeL': generation_metrics.get('rougeL', 0.0),
        })
    
    summary_path = output_path / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"\nClassification Metrics:")
    print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {classification_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {classification_metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro): {classification_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {classification_metrics['recall_macro']:.4f}")
    
    if generation_metrics:
        print(f"\nGeneration Metrics:")
        print(f"  BLEU: {generation_metrics.get('bleu', 0.0):.4f}")
        print(f"  ROUGE-1: {generation_metrics.get('rouge1', 0.0):.4f}")
        print(f"  ROUGE-L: {generation_metrics.get('rougeL', 0.0):.4f}")
    
    print(f"\nResults saved to: {output_path}")
    
    return evaluation_results


# ============================================
# CLI Interface
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VetLLM Evaluation Pipeline")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model (adapter or full model)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test.json file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (required if model-path is an adapter)"
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Model is a full model, not an adapter"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples for quick evaluation"
    )
    parser.add_argument(
        "--no-generation-metrics",
        action="store_true",
        help="Skip BLEU/ROUGE computation"
    )
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        is_adapter=not args.no_adapter,
        num_samples=args.num_samples,
        compute_generation_metrics=not args.no_generation_metrics,
    )
    
    evaluate(config)

