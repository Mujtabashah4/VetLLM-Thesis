#!/usr/bin/env python3
"""
Apply Oversampling to Training Data
Oversamples rare diseases to balance the dataset further
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict

def oversample_rare_diseases(
    input_path: str,
    output_path: str,
    target_samples: int = 30,
    min_samples_threshold: int = 30
):
    """Oversample rare diseases in training data."""
    
    print("="*80)
    print("OVERSAMPLING RARE DISEASES")
    print("="*80)
    
    # Load data
    print(f"\n1. Loading data from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"   Original samples: {len(data)}")
    
    # Analyze distribution
    print("\n2. Analyzing disease distribution...")
    disease_counts = Counter()
    disease_samples = {}
    
    for item in data:
        metadata = item.get('metadata', {})
        disease = metadata.get('disease_normalized') or metadata.get('disease', 'Unknown')
        if disease and disease != 'Unknown':
            disease_counts[disease] += 1
            if disease not in disease_samples:
                disease_samples[disease] = []
            disease_samples[disease].append(item)
    
    # Identify rare diseases
    rare_diseases = {
        disease: count 
        for disease, count in disease_counts.items() 
        if count < min_samples_threshold
    }
    
    print(f"\n3. Rare diseases identified (<{min_samples_threshold} samples): {len(rare_diseases)}")
    
    # Oversample
    print("\n4. Oversampling rare diseases...")
    oversampled_data = list(data)  # Start with all original data
    
    for disease, current_count in rare_diseases.items():
        if disease not in disease_samples:
            continue
            
        needed = target_samples - current_count
        if needed > 0:
            samples = disease_samples[disease]
            # Repeat samples to reach target
            repetitions = (needed // len(samples)) + 1
            for _ in range(repetitions):
                oversampled_data.extend(random.sample(samples, min(needed, len(samples))))
                if len([x for x in oversampled_data if x.get('metadata', {}).get('disease_normalized') == disease]) >= target_samples:
                    break
            
            new_count = len([x for x in oversampled_data if x.get('metadata', {}).get('disease_normalized') == disease])
            print(f"   {disease:40s} {current_count:3d} → {new_count:3d} (+{new_count - current_count})")
    
    # Filter to target counts
    final_data = []
    disease_added = Counter()
    
    for item in oversampled_data:
        metadata = item.get('metadata', {})
        disease = metadata.get('disease_normalized') or metadata.get('disease', 'Unknown')
        
        if disease in rare_diseases:
            if disease_added[disease] < target_samples:
                final_data.append(item)
                disease_added[disease] += 1
        else:
            final_data.append(item)
            disease_added[disease] += 1
    
    # Save
    print(f"\n5. Saving oversampled dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    # Final stats
    print("\n" + "="*80)
    print("OVERSAMPLING SUMMARY")
    print("="*80)
    print(f"Original samples:     {len(data)}")
    print(f"Oversampled samples:  {len(final_data)}")
    print(f"Increase:            {len(final_data) - len(data)} samples")
    
    # Verify
    final_counts = Counter()
    for item in final_data:
        metadata = item.get('metadata', {})
        disease = metadata.get('disease_normalized') or metadata.get('disease', 'Unknown')
        if disease and disease != 'Unknown':
            final_counts[disease] += 1
    
    print("\n6. Final distribution (rare diseases):")
    print("   " + "-"*76)
    for disease in rare_diseases.keys():
        old_count = disease_counts.get(disease, 0)
        new_count = final_counts.get(disease, 0)
        print(f"   {disease:40s} {old_count:3d} → {new_count:3d}")
    
    print("\n" + "="*80)
    print("✅ OVERSAMPLING COMPLETE!")
    print("="*80)
    
    return final_data


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "experiments" / "qwen2.5-7b" / "data" / "train_augmented.json"
    output_path = base_dir / "experiments" / "qwen2.5-7b" / "data" / "train_balanced.json"
    
    oversampled_data = oversample_rare_diseases(
        input_path=str(input_path),
        output_path=str(output_path),
        target_samples=30,
        min_samples_threshold=30
    )
    
    print(f"\n✅ Balanced dataset saved to: {output_path}")

