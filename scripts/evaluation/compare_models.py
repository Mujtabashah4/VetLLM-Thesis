#!/usr/bin/env python3
"""
Model Comparison Script
Compares QWEN and Alpaca-7b fine-tuned models on the same test set
"""

import json
from pathlib import Path
from typing import Dict, Any

def load_evaluation_results(qwen_path: str, alpaca_path: str) -> Dict[str, Any]:
    """Load evaluation results from both models."""
    results = {}
    
    # Load QWEN results
    if Path(qwen_path).exists():
        with open(qwen_path, 'r') as f:
            results['qwen'] = json.load(f)
    else:
        results['qwen'] = None
    
    # Load Alpaca results
    if Path(alpaca_path).exists():
        with open(alpaca_path, 'r') as f:
            results['alpaca'] = json.load(f)
    else:
        results['alpaca'] = None
    
    return results

def print_comparison(results: Dict[str, Any]):
    """Print comparison table."""
    print("="*80)
    print("MODEL COMPARISON: QWEN vs ALPACA-7B")
    print("="*80)
    
    qwen_metrics = results.get('qwen', {}).get('metrics', {}) if results.get('qwen') else None
    alpaca_metrics = results.get('alpaca', {}).get('metrics', {}) if results.get('alpaca') else None
    
    if not qwen_metrics:
        print("⚠️  QWEN results not available yet")
        return
    
    print("\n📊 Disease Classification Metrics:")
    print("-" * 80)
    print(f"{'Metric':<25} {'QWEN':<20} {'Alpaca-7b':<20}")
    print("-" * 80)
    
    disease_metrics = qwen_metrics.get('disease_classification', {})
    alpaca_disease = alpaca_metrics.get('disease_classification', {}) if alpaca_metrics else {}
    
    metrics_to_show = [
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Precision (Macro)'),
        ('recall_macro', 'Recall (Macro)'),
        ('f1_macro', 'F1 Score (Macro)'),
        ('f1_micro', 'F1 Score (Micro)'),
        ('f1_weighted', 'F1 Score (Weighted)'),
    ]
    
    for key, label in metrics_to_show:
        qwen_val = disease_metrics.get(key, 0.0)
        alpaca_val = alpaca_disease.get(key, 0.0) if alpaca_metrics else None
        
        if alpaca_val is not None:
            diff = qwen_val - alpaca_val
            diff_str = f"({diff:+.4f})" if diff != 0 else ""
            print(f"{label:<25} {qwen_val:<20.4f} {alpaca_val:<20.4f} {diff_str}")
        else:
            print(f"{label:<25} {qwen_val:<20.4f} {'N/A':<20}")
    
    print("\n🏷️  SNOMED Code Metrics:")
    print("-" * 80)
    qwen_snomed = qwen_metrics.get('snomed_code', {})
    alpaca_snomed = alpaca_metrics.get('snomed_code', {}) if alpaca_metrics else {}
    
    print(f"{'Metric':<25} {'QWEN':<20} {'Alpaca-7b':<20}")
    print("-" * 80)
    print(f"{'Accuracy':<25} {qwen_snomed.get('accuracy', 0.0):<20.4f} {alpaca_snomed.get('accuracy', 0.0) if alpaca_metrics else 'N/A':<20}")
    print(f"{'Correct/Total':<25} {qwen_snomed.get('correct', 0)}/{qwen_snomed.get('total', 0):<19} {f\"{alpaca_snomed.get('correct', 0)}/{alpaca_snomed.get('total', 0)}\" if alpaca_metrics else 'N/A':<20}")
    
    print("\n" + "="*80)

def main():
    """Main comparison function."""
    qwen_results_path = "reports/qwen_comprehensive_evaluation.json"
    alpaca_results_path = "reports/comprehensive_validation_results.json"
    
    results = load_evaluation_results(qwen_results_path, alpaca_results_path)
    
    print_comparison(results)
    
    # Save comparison
    output_path = "reports/model_comparison.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {
        'qwen_available': results['qwen'] is not None,
        'alpaca_available': results['alpaca'] is not None,
        'qwen_metrics': results['qwen'].get('metrics', {}) if results['qwen'] else {},
        'alpaca_metrics': results['alpaca'].get('metrics', {}) if results['alpaca'] else {},
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comparison saved to: {output_path}")

if __name__ == "__main__":
    main()

