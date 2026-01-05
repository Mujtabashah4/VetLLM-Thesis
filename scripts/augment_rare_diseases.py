#!/usr/bin/env python3
"""
Data Augmentation Script for Rare Diseases
Generates synthetic training examples for diseases with <10 samples to address class imbalance
"""

import json
import random
import copy
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# Symptom combinations for each disease (based on veterinary knowledge)
DISEASE_SYMPTOM_PATTERNS = {
    "Anthrax": [
        ["Fever(F)", "Fluid leakage from nose", "Sudden death"],
        ["Fever(F)", "Blood leakage from nose", "Very high fever"],
        ["Fever(F)", "Neck swelling", "Difficulty in breathing", "Sudden death"],
        ["High fever", "Nasal discharge", "Bloody diarrhea"],
    ],
    "Black Quarter": [
        ["Color change of muscles of limb", "Lameness", "Sudden death"],
        ["Stiffening of body", "Lameness", "Muscle discoloration"],
        ["Lameness", "Swollen muscles", "Dark colored muscles"],
    ],
    "Contagious Caprine Pleuropneumonia": [
        ["Cough", "Difficulty in breathing", "Rapid respiratory rate", "Fever"],
        ["Persistent cough", "Breathing while mouth open", "Nasal discharge"],
        ["Severe cough", "Difficulty breathing", "Fever", "Nasal discharge"],
    ],
    "Brucellosis": [
        ["Abortion", "Vaginal gnarls", "Drop in milk production"],
        ["Abortion", "Weakness", "Loss of appetite"],
        ["Abortion", "Fever", "Joint swelling"],
    ],
    "Babesiosis": [
        ["High fever", "Pale mucous membranes", "Weakness", "Anemia"],
        ["Fever", "Weakness", "Loss of appetite", "Pale membranes"],
        ["High fever", "Weakness", "Anemia", "Dark urine"],
    ],
    "Theileriosis": [
        ["High fever", "Weakness", "Loss of appetite", "Pale mucous membranes"],
        ["Fever", "Weakness", "Anemia", "Enlarged lymph nodes"],
        ["Very high fever", "Weakness", "Pale membranes", "Weight loss"],
    ],
    "Rabies": [
        ["Aggressive behavior", "Excessive salivation", "Difficulty swallowing", "Paralysis"],
        ["Behavioral changes", "Excessive salivation", "Paralysis", "Hydrophobia"],
        ["Aggression", "Salivation", "Paralysis", "Difficulty swallowing"],
    ],
    "Liver Fluke": [
        ["Chronic diarrhea", "Weight loss", "Anemia", "Weakness"],
        ["Diarrhea", "Weight loss", "Pale membranes", "Weakness"],
        ["Loose motions", "Emaciation", "Anemia", "Weakness"],
    ],
    "Internal Worms": [
        ["Weakness", "Loss of appetite", "Pale mucous membranes", "Weight loss"],
        ["Weakness", "Anemia", "Weight loss", "Poor growth"],
        ["Weakness", "Diarrhea", "Anemia", "Weight loss"],
    ],
    "Foot Rot": [
        ["Lameness", "Swollen feet", "Difficulty walking"],
        ["Lameness", "Foot swelling", "Pain in feet"],
        ["Lameness", "Swollen hooves", "Difficulty walking"],
    ],
    "Ketosis": [
        ["Drop in milk production", "Weakness", "Loss of appetite"],
        ["Ketosis", "Drop in milk", "Weakness"],
        ["Milk production drop", "Weakness", "Ketone smell"],
    ],
    "Tympany": [
        ["Tympany (bloated abdomen)", "Pain in stomach", "Difficulty breathing"],
        ["Bloated abdomen", "Stomach pain", "Rapid breathing"],
        ["Bloat", "Abdominal distension", "Difficulty breathing"],
    ],
    "Fracture of the Leg": [
        ["Fracture of the leg", "Lameness", "Inability to bear weight"],
        ["Leg fracture", "Lameness", "Refusal to bear weight"],
        ["Fracture", "Lameness", "Unable to walk"],
    ],
    "Laminitis": [
        ["Lameness", "Pain in feet", "Difficulty walking"],
        ["Foot pain", "Lameness", "Reluctance to move"],
        ["Lameness", "Hot feet", "Pain"],
    ],
    "Flue": [
        ["Fever", "Cough", "Nasal discharge"],
        ["Fever", "Cough", "Weakness"],
        ["Fever", "Respiratory symptoms", "Nasal discharge"],
    ],
    "Goat Pox": [
        ["Fever", "Skin lesions", "Pox lesions"],
        ["Fever", "Skin nodules", "Lesions"],
        ["Fever", "Pox", "Skin abnormalities"],
    ],
    "Abortion": [
        ["Abortion", "Vaginal discharge", "Weakness"],
        ["Abortion", "Fever", "Loss of appetite"],
        ["Abortion", "Vaginal signs", "Weakness"],
    ],
    "Infection": [
        ["Fever", "Weakness", "Loss of appetite"],
        ["Fever", "Generalized symptoms", "Weakness"],
        ["Fever", "Infection signs", "Weakness"],
    ],
}

# Animal species for each disease (based on veterinary knowledge)
DISEASE_SPECIES = {
    "Anthrax": ["Cow", "Buffalo", "Sheep", "Goat"],
    "Black Quarter": ["Cow", "Buffalo"],
    "Contagious Caprine Pleuropneumonia": ["Goat"],
    "Brucellosis": ["Cow", "Buffalo", "Goat"],
    "Babesiosis": ["Cow", "Buffalo"],
    "Theileriosis": ["Cow", "Buffalo"],
    "Rabies": ["Cow", "Buffalo", "Sheep", "Goat"],
    "Liver Fluke": ["Sheep", "Goat"],
    "Internal Worms": ["Sheep", "Goat", "Cow"],
    "Foot Rot": ["Sheep", "Goat"],
    "Ketosis": ["Cow"],
    "Tympany": ["Cow", "Buffalo", "Sheep", "Goat"],
    "Fracture of the Leg": ["Cow", "Buffalo", "Sheep", "Goat"],
    "Laminitis": ["Cow", "Horse"],
    "Flue": ["Cow", "Buffalo", "Sheep", "Goat"],
    "Goat Pox": ["Goat"],
    "Abortion": ["Cow", "Buffalo", "Goat"],
    "Infection": ["Cow", "Buffalo", "Sheep", "Goat"],
}

# SNOMED codes for diseases
SNOMED_CODES = {
    "Anthrax": ["40214000"],
    "Black Quarter": ["29600000"],
    "Contagious Caprine Pleuropneumonia": ["2260006"],
    "Brucellosis": ["75702008"],
    "Babesiosis": ["24026003"],
    "Theileriosis": ["24694002"],
    "Rabies": ["14146002"],
    "Liver Fluke": ["4764006"],
    "Internal Worms": [],
    "Foot Rot": [],
    "Ketosis": [],
    "Tympany": [],
    "Fracture of the Leg": [],
    "Laminitis": [],
    "Flue": [],
    "Goat Pox": [],
    "Abortion": [],
    "Infection": [],
}


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_disease_distribution(data: List[Dict]) -> Dict[str, int]:
    """Analyze disease distribution in training data."""
    disease_counts = Counter()
    
    for item in data:
        # Extract disease from metadata
        metadata = item.get('metadata', {})
        disease = metadata.get('disease_normalized') or metadata.get('disease', 'Unknown')
        if disease and disease != 'Unknown':
            disease_counts[disease] += 1
    
    return dict(disease_counts)


def create_augmented_sample(
    disease: str,
    symptoms: List[str],
    species: str,
    snomed_codes: List[str],
    base_sample: Dict = None
) -> Dict:
    """Create an augmented training sample."""
    
    # Create symptom list for clinical signs
    clinical_signs = ", ".join(symptoms).lower()
    
    # Create input text
    input_text = f"Analyze this veterinary case:\n\n**Species**: {species}\n**Clinical Signs**: {clinical_signs}\n\nProvide your diagnosis, differentials, treatment plan, and reasoning."
    
    # Create output text
    output_text = f"1. **Primary Diagnosis**: **{disease}**"
    if snomed_codes:
        output_text += f" (SNOMED-CT: {', '.join(snomed_codes)})"
    output_text += f"\n\n2. **Differential Diagnoses**:\n   - Consider other diseases with similar symptoms\n\n3. **Recommended Treatment**:\nConsult with veterinarian for appropriate treatment protocol.\n\n4. **Clinical Reasoning**:\nBased on the presenting clinical signs ({', '.join(symptoms)}), the pattern is consistent with {disease}. The combination of symptoms and species-specific epidemiology supports this diagnosis."
    
    # Create full text with system prompt
    system_prompt = "You are VetLLM, a veterinary clinical assistant specialized in livestock diseases. Given a clinical case presentation, you will:\n1. Analyze the clinical signs and symptoms\n2. Provide the most likely diagnosis with SNOMED-CT codes when available\n3. List differential diagnoses to consider\n4. Recommend appropriate treatment and management steps\n5. Explain your clinical reasoning"
    
    full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
    
    # Create sample structure
    sample = {
        "text": full_text,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "animal": species,
            "disease": disease,
            "disease_normalized": disease,
            "snomed_codes": snomed_codes,
            "symptoms": symptoms,
            "augmented": True  # Mark as augmented
        }
    }
    
    return sample


def generate_augmented_samples(
    disease: str,
    target_count: int,
    current_count: int
) -> List[Dict]:
    """Generate augmented samples for a disease."""
    samples = []
    needed = max(0, target_count - current_count)
    
    if needed == 0:
        return samples
    
    # Get symptom patterns for this disease
    patterns = DISEASE_SYMPTOM_PATTERNS.get(disease, [])
    if not patterns:
        print(f"⚠️  Warning: No symptom patterns defined for {disease}")
        return samples
    
    # Get species for this disease
    species_list = DISEASE_SPECIES.get(disease, ["Cow", "Buffalo", "Sheep", "Goat"])
    
    # Get SNOMED codes
    snomed_codes = SNOMED_CODES.get(disease, [])
    
    # Generate samples
    for i in range(needed):
        # Randomly select symptom pattern
        symptom_pattern = random.choice(patterns)
        
        # Add some variation (randomly add/remove 1-2 symptoms)
        symptoms = copy.copy(symptom_pattern)
        if random.random() < 0.3:  # 30% chance to add variation
            all_symptoms = [
                "Fever(F)", "Fluid leakage from nose", "Loose motions", "Cough",
                "Blisters on lips", "Lameness", "Stiffening of body", "Nasal discharge",
                "Severe cough", "Pain in stomach", "Bloody diarrhea", "Blood in milk",
                "Vaginal abnormalities", "Teat abnormalities", "Weakness"
            ]
            # Add 1-2 random symptoms (if not already present)
            for _ in range(random.randint(1, 2)):
                new_symptom = random.choice(all_symptoms)
                if new_symptom not in symptoms:
                    symptoms.append(new_symptom)
        
        # Randomly select species
        species = random.choice(species_list)
        
        # Create sample
        sample = create_augmented_sample(
            disease=disease,
            symptoms=symptoms,
            species=species,
            snomed_codes=snomed_codes
        )
        
        samples.append(sample)
    
    return samples


def augment_training_data(
    input_path: str,
    output_path: str,
    target_samples_per_disease: int = 25,
    min_samples_threshold: int = 10
):
    """Augment training data for rare diseases."""
    
    print("="*80)
    print("DATA AUGMENTATION FOR RARE DISEASES")
    print("="*80)
    
    # Load original training data
    print(f"\n1. Loading training data from: {input_path}")
    original_data = load_training_data(input_path)
    print(f"   Original samples: {len(original_data)}")
    
    # Analyze disease distribution
    print("\n2. Analyzing disease distribution...")
    disease_counts = analyze_disease_distribution(original_data)
    
    print("\n   Disease Distribution:")
    print("   " + "-"*76)
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    for disease, count in sorted_diseases:
        status = "✅" if count >= min_samples_threshold else "❌"
        print(f"   {status} {disease:40s} {count:4d} samples")
    
    # Identify rare diseases
    rare_diseases = {
        disease: count 
        for disease, count in disease_counts.items() 
        if count < min_samples_threshold
    }
    
    print(f"\n3. Rare diseases identified (<{min_samples_threshold} samples): {len(rare_diseases)}")
    
    # Generate augmented samples
    print("\n4. Generating augmented samples...")
    augmented_samples = []
    
    for disease, current_count in rare_diseases.items():
        needed = target_samples_per_disease - current_count
        if needed > 0:
            print(f"   Generating {needed} samples for {disease}...")
            new_samples = generate_augmented_samples(
                disease=disease,
                target_count=target_samples_per_disease,
                current_count=current_count
            )
            augmented_samples.extend(new_samples)
            print(f"   ✅ Generated {len(new_samples)} samples for {disease}")
    
    # Combine original and augmented data
    print("\n5. Combining original and augmented data...")
    combined_data = original_data + augmented_samples
    
    # Save augmented dataset
    print(f"\n6. Saving augmented dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    # Final statistics
    print("\n" + "="*80)
    print("AUGMENTATION SUMMARY")
    print("="*80)
    print(f"Original samples:     {len(original_data)}")
    print(f"Augmented samples:   {len(augmented_samples)}")
    print(f"Total samples:       {len(combined_data)}")
    print(f"Augmentation ratio:  {len(augmented_samples)/len(original_data)*100:.1f}%")
    
    # Verify new distribution
    print("\n7. Verifying new disease distribution...")
    new_disease_counts = analyze_disease_distribution(combined_data)
    
    print("\n   Updated Distribution (rare diseases):")
    print("   " + "-"*76)
    for disease in rare_diseases.keys():
        old_count = disease_counts.get(disease, 0)
        new_count = new_disease_counts.get(disease, 0)
        improvement = new_count - old_count
        status = "✅" if new_count >= min_samples_threshold else "⚠️"
        print(f"   {status} {disease:40s} {old_count:3d} → {new_count:3d} (+{improvement})")
    
    print("\n" + "="*80)
    print("✅ DATA AUGMENTATION COMPLETE!")
    print("="*80)
    
    return combined_data, augmented_samples


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "experiments" / "qwen2.5-7b" / "data" / "train.json"
    output_path = base_dir / "experiments" / "qwen2.5-7b" / "data" / "train_augmented.json"
    
    # Create backup of original
    backup_path = base_dir / "experiments" / "qwen2.5-7b" / "data" / "train_original.json"
    if not backup_path.exists():
        print(f"\nCreating backup: {backup_path}")
        import shutil
        shutil.copy(input_path, backup_path)
    
    # Run augmentation
    augmented_data, new_samples = augment_training_data(
        input_path=str(input_path),
        output_path=str(output_path),
        target_samples_per_disease=25,
        min_samples_threshold=10
    )
    
    print(f"\n✅ Augmented dataset saved to: {output_path}")
    print(f"   Ready for training with improved class balance!")

