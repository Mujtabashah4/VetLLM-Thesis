#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for QWEN Fine-tuned Model
Computes F1, Precision, Recall, Accuracy and other metrics for publication-ready results
"""

import json
import re
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add shared inference to path
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "shared"))
from inference import ModelConfig, load_model, generate_diagnosis, ChatFormatter


class QwenComprehensiveEvaluator:
    """Comprehensive evaluator for QWEN model with full metrics."""
    
    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.snomed_mapping = self.load_snomed_mapping()
        
    def load_snomed_mapping(self):
        """Load SNOMED code mappings."""
        mapping_path = Path("snomed_mapping.json")
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_model(self):
        """Load the fine-tuned model."""
        print("="*80)
        print("Loading QWEN Fine-tuned Model...")
        print("="*80)
        
        config = ModelConfig(
            model_name_or_path=self.base_model_path,
            adapter_path=self.adapter_path,
            use_4bit=True,
            use_flash_attention=False,
        )
        
        self.model, self.tokenizer = load_model(config)
        print("✅ Model loaded successfully!\n")
    
    def extract_snomed_codes(self, text: str) -> List[str]:
        """Extract SNOMED-CT codes from text."""
        # Pattern for SNOMED codes (7-8 digits)
        codes = re.findall(r'\b(\d{7,8})\b', text)
        # Also check for codes in parentheses or after colons
        codes.extend(re.findall(r'(?:SNOMED-CT|SNOMED|code)[\s:]+(\d{7,8})', text, re.IGNORECASE))
        codes.extend(re.findall(r'\((\d{7,8})\)', text))
        
        # Get valid codes from mapping
        valid_codes = set()
        for disease, codes_list in self.snomed_mapping.get('diseases', {}).items():
            valid_codes.update(codes_list)
        
        # Filter to valid codes
        extracted = [c for c in codes if c in valid_codes]
        
        # If no valid codes found, return all numeric codes found
        return extracted if extracted else list(set(codes))[:5]
    
    def extract_disease_name(self, text: str) -> str:
        """Extract disease name from model output."""
        text_lower = text.lower()
        
        # Disease name patterns
        disease_patterns = {
            'peste des petits ruminants': 'Peste des Petits Ruminants',
            'ppr': 'Peste des Petits Ruminants',
            'p.p.r': 'Peste des Petits Ruminants',
            'anthrax': 'Anthrax',
            'foot and mouth disease': 'Foot and Mouth Disease',
            'fmd': 'Foot and Mouth Disease',
            'hemorrhagic septicemia': 'Hemorrhagic Septicemia',
            'h.s': 'Hemorrhagic Septicemia',
            'mastitis': 'Mastitis',
            'mastits': 'Mastitis',
            'black quarter': 'Black Quarter',
            'b.q': 'Black Quarter',
            'contagious caprine pleuropneumonia': 'Contagious Caprine Pleuropneumonia',
            'ccpp': 'Contagious Caprine Pleuropneumonia',
            'brucellosis': 'Brucellosis',
            'babesiosis': 'Babesiosis',
            'theileriosis': 'Theileriosis',
            'rabies': 'Rabies',
            'foot rot': 'Foot Rot',
            'liver fluke': 'Liver Fluke',
        }
        
        for pattern, disease in disease_patterns.items():
            if pattern in text_lower:
                return disease
        
        return 'Unknown'
    
    def normalize_disease(self, disease: str) -> str:
        """Normalize disease name for comparison."""
        mapping = {
            'PPR': 'Peste des Petits Ruminants',
            'P.P.R': 'Peste des Petits Ruminants',
            'FMD': 'Foot and Mouth Disease',
            'H.S': 'Hemorrhagic Septicemia',
            'B.Q': 'Black Quarter',
            'CCPP': 'Contagious Caprine Pleuropneumonia',
            'Mastits': 'Mastitis',
        }
        return mapping.get(disease, disease)
    
    def evaluate_on_test_set(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluate model on test set."""
        print("="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(test_data)} test samples\n")
        
        # Prepare predictions
        predictions = []
        references = []
        snomed_predictions = []
        snomed_references = []
        
        results = []
        
        for i, sample in enumerate(test_data, 1):
            if i % 10 == 0:
                print(f"Processing {i}/{len(test_data)}...")
            
            # Extract ground truth
            metadata = sample.get('metadata', {})
            expected_disease = metadata.get('disease_normalized', 'Unknown')
            expected_snomed = metadata.get('snomed_codes', [])
            
            # Get input
            input_text = sample.get('input', '')
            if not input_text:
                # Extract from text field
                text = sample.get('text', '')
                # Extract user message
                user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
                if user_match:
                    input_text = user_match.group(1).strip()
            
            # Extract species and symptoms from input
            species_match = re.search(r'\*\*Species\*\*:\s*(\w+)', input_text)
            symptoms_match = re.search(r'\*\*Clinical Signs\*\*:\s*(.+?)(?:\n|$)', input_text)
            
            species = species_match.group(1) if species_match else metadata.get('animal', 'Unknown')
            symptoms = symptoms_match.group(1) if symptoms_match else ', '.join(metadata.get('symptoms', []))
            
            # Generate prediction
            try:
                response = generate_diagnosis(
                    self.model, self.tokenizer,
                    species=species,
                    symptoms=symptoms,
                    model_type="qwen2.5",
                )
            except Exception as e:
                print(f"Error generating prediction for sample {i}: {e}")
                response = ""
            
            # Extract predicted disease and codes
            predicted_disease = self.extract_disease_name(response)
            predicted_snomed = self.extract_snomed_codes(response)
            
            # Normalize
            expected_disease_norm = self.normalize_disease(expected_disease)
            predicted_disease_norm = self.normalize_disease(predicted_disease)
            
            # Store
            predictions.append(predicted_disease_norm)
            references.append(expected_disease_norm)
            snomed_predictions.append(predicted_snomed)
            snomed_references.append(expected_snomed)
            
            results.append({
                'sample_id': i,
                'species': species,
                'symptoms': symptoms,
                'expected_disease': expected_disease_norm,
                'predicted_disease': predicted_disease_norm,
                'expected_snomed': expected_snomed,
                'predicted_snomed': predicted_snomed,
                'response': response,
                'correct_disease': expected_disease_norm == predicted_disease_norm,
                'correct_snomed': any(code in predicted_snomed for code in expected_snomed) if expected_snomed else False,
            })
        
        print(f"\n✅ Completed evaluation on {len(test_data)} samples\n")
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            predictions, references,
            snomed_predictions, snomed_references
        )
        
        return {
            'metrics': metrics,
            'results': results,
            'predictions': predictions,
            'references': references,
        }
    
    def calculate_metrics(
        self,
        disease_predictions: List[str],
        disease_references: List[str],
        snomed_predictions: List[List[str]],
        snomed_references: List[List[str]],
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        
        # Disease classification metrics
        unique_labels = sorted(list(set(disease_references + disease_predictions)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        y_true = [label_to_id.get(r, -1) for r in disease_references]
        y_pred = [label_to_id.get(p, -1) for p in disease_predictions]
        
        # Filter valid predictions
        valid_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != -1 and p != -1]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        # Calculate disease metrics
        disease_accuracy = accuracy_score(y_true_valid, y_pred_valid) if valid_indices else 0.0
        disease_precision = precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) if valid_indices else 0.0
        disease_recall = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) if valid_indices else 0.0
        disease_f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) if valid_indices else 0.0
        disease_f1_micro = f1_score(y_true_valid, y_pred_valid, average='micro', zero_division=0) if valid_indices else 0.0
        disease_f1_weighted = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0) if valid_indices else 0.0
        
        # SNOMED code metrics
        snomed_correct = 0
        snomed_total = 0
        for pred_codes, ref_codes in zip(snomed_predictions, snomed_references):
            if ref_codes:
                snomed_total += 1
                if any(code in pred_codes for code in ref_codes):
                    snomed_correct += 1
        
        snomed_accuracy = (snomed_correct / snomed_total) if snomed_total > 0 else 0.0
        
        # Per-disease metrics
        per_disease_metrics = {}
        for disease in unique_labels:
            if disease == 'Unknown':
                continue
            disease_indices = [i for i, ref in enumerate(disease_references) if ref == disease]
            if not disease_indices:
                continue
            
            disease_correct = sum(1 for i in disease_indices if disease_predictions[i] == disease)
            per_disease_metrics[disease] = {
                'total': len(disease_indices),
                'correct': disease_correct,
                'accuracy': disease_correct / len(disease_indices) if disease_indices else 0.0,
            }
        
        return {
            'disease_classification': {
                'accuracy': disease_accuracy,
                'precision_macro': disease_precision,
                'recall_macro': disease_recall,
                'f1_macro': disease_f1_macro,
                'f1_micro': disease_f1_micro,
                'f1_weighted': disease_f1_weighted,
            },
            'snomed_code': {
                'accuracy': snomed_accuracy,
                'correct': snomed_correct,
                'total': snomed_total,
            },
            'per_disease': per_disease_metrics,
            'total_samples': len(disease_references),
        }
    
    def print_results(self, evaluation_results: Dict[str, Any]):
        """Print comprehensive results."""
        metrics = evaluation_results['metrics']
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        print("\n📊 Disease Classification Metrics:")
        print("-" * 80)
        disease_metrics = metrics['disease_classification']
        print(f"  Accuracy:        {disease_metrics['accuracy']:.4f} ({disease_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision (Macro): {disease_metrics['precision_macro']:.4f} ({disease_metrics['precision_macro']*100:.2f}%)")
        print(f"  Recall (Macro):    {disease_metrics['recall_macro']:.4f} ({disease_metrics['recall_macro']*100:.2f}%)")
        print(f"  F1 Score (Macro):  {disease_metrics['f1_macro']:.4f} ({disease_metrics['f1_macro']*100:.2f}%)")
        print(f"  F1 Score (Micro):  {disease_metrics['f1_micro']:.4f} ({disease_metrics['f1_micro']*100:.2f}%)")
        print(f"  F1 Score (Weighted): {disease_metrics['f1_weighted']:.4f} ({disease_metrics['f1_weighted']*100:.2f}%)")
        
        print("\n🏷️  SNOMED Code Metrics:")
        print("-" * 80)
        snomed_metrics = metrics['snomed_code']
        print(f"  Accuracy: {snomed_metrics['accuracy']:.4f} ({snomed_metrics['accuracy']*100:.2f}%)")
        print(f"  Correct:  {snomed_metrics['correct']}/{snomed_metrics['total']}")
        
        print("\n📋 Per-Disease Performance:")
        print("-" * 80)
        print(f"{'Disease':<35} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
        print("-" * 80)
        for disease, stats in sorted(metrics['per_disease'].items()):
            print(f"{disease:<35} {stats['total']:<10} {stats['correct']:<10} {stats['accuracy']*100:<10.2f}%")
        
        print("\n" + "="*80)
    
    def save_results(self, evaluation_results: Dict[str, Any], output_path: str):
        """Save results to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main evaluation function."""
    base_model = "/home/iml_admin/Desktop/VetLLM/VetLLM-Thesis/models/qwen2.5-7b-instruct"
    adapter_path = "/home/iml_admin/Desktop/VetLLM/VetLLM-Thesis/experiments/qwen2.5-7b/checkpoints/final"
    test_data_path = "experiments/qwen2.5-7b/data/test.json"
    output_path = "reports/qwen_comprehensive_evaluation.json"
    
    print("="*80)
    print("QWEN COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Base Model: {base_model}")
    print(f"Adapter: {adapter_path}")
    print(f"Test Data: {test_data_path}")
    print("="*80)
    
    evaluator = QwenComprehensiveEvaluator(base_model, adapter_path)
    evaluator.load_model()
    
    # Evaluate
    results = evaluator.evaluate_on_test_set(test_data_path)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, output_path)
    
    print("\n✅ Comprehensive evaluation complete!")


if __name__ == "__main__":
    main()

