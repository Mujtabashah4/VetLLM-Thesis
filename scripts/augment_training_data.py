#!/usr/bin/env python3
"""
Script to augment training data for rare diseases and small animals.
Generates additional training examples for diseases with low sample counts.
"""
import json
import random
from pathlib import Path
from typing import List, Dict

# Rare diseases that need more data
RARE_DISEASES = {
    "CCPP": {
        "symptoms": [
            "severe cough", "difficulty breathing", "rapid breathing", "fever",
            "persistent cough", "breathing while mouth open", "nasal discharge",
            "respiratory distress", "labored breathing", "coughing fits"
        ],
        "animals": ["Goat", "Sheep"],
        "snomed_code": "2260006"
    },
    "Brucellosis": {
        "symptoms": [
            "abortion", "vaginal gnarls", "drop in milk production",
            "reduced milk yield", "retained placenta", "infertility",
            "swollen joints", "lameness", "testicular swelling"
        ],
        "animals": ["Cow", "Buffalo", "Goat", "Sheep"],
        "snomed_code": "75702008"
    },
    "Babesiosis": {
        "symptoms": [
            "high fever", "pale mucous membranes", "weakness", "anemia",
            "jaundice", "dark urine", "loss of appetite", "rapid breathing",
            "increased heart rate", "depression"
        ],
        "animals": ["Cow", "Buffalo"],
        "snomed_code": "24026003"
    },
    "Theileriosis": {
        "symptoms": [
            "high fever", "weakness", "loss of appetite", "pale mucous membranes",
            "enlarged lymph nodes", "jaundice", "anemia", "depression",
            "reduced milk production", "swollen eyes"
        ],
        "animals": ["Cow", "Buffalo"],
        "snomed_code": "24694002"
    },
    "Rabies": {
        "symptoms": [
            "aggressive behavior", "excessive salivation", "difficulty swallowing",
            "paralysis", "abnormal behavior", "hydrophobia", "muscle spasms",
            "incoordination", "drooping jaw", "vocalization changes"
        ],
        "animals": ["Goat", "Sheep", "Cow", "Buffalo"],
        "snomed_code": "14146002"
    },
    "Liver Fluke": {
        "symptoms": [
            "chronic diarrhea", "weight loss", "anemia", "weakness",
            "bottle jaw", "pale mucous membranes", "reduced milk production",
            "poor body condition", "lethargy", "jaundice"
        ],
        "animals": ["Sheep", "Goat", "Cow"],
        "snomed_code": "4764006"
    },
    "Internal Worms": {
        "symptoms": [
            "weakness", "loss of appetite", "pale mucous membranes", "weight loss",
            "diarrhea", "poor growth", "pot belly", "rough coat",
            "anemia", "reduced milk production"
        ],
        "animals": ["Goat", "Sheep", "Cow"],
        "snomed_code": None  # No specific code
    },
    "Foot Rot": {
        "symptoms": [
            "lameness", "swollen feet", "difficulty walking", "foot lesions",
            "foul smell from feet", "separation of hoof", "inflammation",
            "discharge from feet", "reluctance to walk", "weight shifting"
        ],
        "animals": ["Sheep", "Goat"],
        "snomed_code": None
    },
    "Mites": {
        "symptoms": [
            "skin lesions", "hair fall", "wool loss", "excessive scratching",
            "restlessness", "thickened skin", "crusts on skin", "hair loss patches",
            "irritation", "rubbing against objects"
        ],
        "animals": ["Sheep", "Goat"],
        "snomed_code": "100000000000120"
    },
    "Ketosis": {
        "symptoms": [
            "drop in milk production", "weakness", "loss of appetite",
            "nervous behavior", "sweet breath", "reduced feed intake",
            "weight loss", "depression", "staggering", "head pressing"
        ],
        "animals": ["Cow"],
        "snomed_code": None
    },
    "Tympany": {
        "symptoms": [
            "bloated abdomen", "pain in stomach", "difficulty breathing",
            "distended left side", "restlessness", "kicking at belly",
            "rapid breathing", "stretching", "groaning", "lying down frequently"
        ],
        "animals": ["Cow", "Goat", "Sheep"],
        "snomed_code": None
    },
    "Fracture": {
        "symptoms": [
            "lameness", "inability to bear weight", "swelling", "pain",
            "abnormal limb position", "crepitus", "reluctance to move",
            "non-weight bearing", "deformity", "guarding behavior"
        ],
        "animals": ["Goat", "Sheep", "Cow"],
        "snomed_code": None
    }
}

# Diseases that need more examples (have some but not enough)
NEED_MORE = {
    "Internal Worms": {"current": 4, "target": 20},
    "Mites": {"current": 2, "target": 15},
    "Ketosis": {"current": 2, "target": 15},
    "Tympany": {"current": 2, "target": 15},
    "Fracture": {"current": 4, "target": 20}
}

def create_clinical_note(animal: str, symptoms: List[str]) -> str:
    """Create a clinical note from animal and symptoms"""
    symptoms_text = ", ".join(symptoms)
    return f"{animal}. Clinical presentation includes {symptoms_text}. Physical examination reveals these clinical signs."

def generate_training_examples(disease: str, disease_info: Dict, num_examples: int = 20) -> List[Dict]:
    """Generate training examples for a disease"""
    examples = []
    snomed_code = disease_info.get("snomed_code")
    animals = disease_info.get("animals", [])
    symptoms_pool = disease_info.get("symptoms", [])
    
    for i in range(num_examples):
        # Randomly select animal
        animal = random.choice(animals)
        
        # Randomly select 3-5 symptoms
        num_symptoms = random.randint(3, min(5, len(symptoms_pool)))
        selected_symptoms = random.sample(symptoms_pool, num_symptoms)
        
        # Create clinical note
        clinical_note = create_clinical_note(animal, selected_symptoms)
        
        # Create output
        if snomed_code:
            output = f"Diagnosed conditions: {snomed_code}"
        else:
            output = f"Diagnosed conditions: {disease}"
        
        # Create training example
        example = {
            "instruction": "Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes.",
            "input": f"Clinical Note: {clinical_note}",
            "output": output,
            "snomed_codes": [snomed_code] if snomed_code else [],
            "disease": disease,
            "animal": animal,
            "symptoms": selected_symptoms
        }
        
        examples.append(example)
    
    return examples

def augment_small_animal_examples(existing_data: List[Dict], target_goat: int = 150, target_sheep: int = 150) -> List[Dict]:
    """Generate more examples for small animals (Goat, Sheep)"""
    # Count current distribution
    goat_count = sum(1 for item in existing_data if "goat" in item.get("input", "").lower() or "goat" in item.get("animal", "").lower())
    sheep_count = sum(1 for item in existing_data if "sheep" in item.get("input", "").lower() or "sheep" in item.get("animal", "").lower())
    
    new_examples = []
    
    # Get diseases common in small animals
    small_animal_diseases = {
        "PPR": {"symptoms": ["fever", "nasal discharge", "coughing", "diarrhea"], "snomed_code": "1679004"},
        "P.P.R": {"symptoms": ["fever", "nasal discharge", "coughing", "diarrhea"], "snomed_code": "1679004"},
        "Kataa": {"symptoms": ["severe respiratory distress", "fever", "continuous loose motions"], "snomed_code": "1679004"},
        "CCPP": {"symptoms": ["severe cough", "difficulty breathing", "rapid breathing", "fever"], "snomed_code": "2260006"},
        "H.S": {"symptoms": ["high fever", "neck swelling", "difficulty breathing", "sudden death"], "snomed_code": "198462004"},
        "Mastitis": {"symptoms": ["swollen udder", "drop in milk production", "milk in semi-solid form"], "snomed_code": "72934000"},
        "Internal Worms": {"symptoms": ["weakness", "loss of appetite", "pale mucous membranes", "weight loss"], "snomed_code": None},
        "Mites": {"symptoms": ["skin lesions", "hair fall", "wool loss", "excessive scratching"], "snomed_code": "100000000000120"},
    }
    
    # Generate Goat examples
    if goat_count < target_goat:
        needed = target_goat - goat_count
        per_disease = max(5, needed // len(small_animal_diseases))
        print(f"  🐐 Goat: Current {goat_count}, generating {needed} more examples")
        for disease, info in small_animal_diseases.items():
            examples = generate_training_examples(disease, {"animals": ["Goat"], "symptoms": info["symptoms"], "snomed_code": info["snomed_code"]}, num_examples=per_disease)
            new_examples.extend([e for e in examples if e["animal"] == "Goat"])
    
    # Generate Sheep examples
    if sheep_count < target_sheep:
        needed = target_sheep - sheep_count
        per_disease = max(5, needed // len(small_animal_diseases))
        print(f"  🐑 Sheep: Current {sheep_count}, generating {needed} more examples")
        for disease, info in small_animal_diseases.items():
            examples = generate_training_examples(disease, {"animals": ["Sheep"], "symptoms": info["symptoms"], "snomed_code": info["snomed_code"]}, num_examples=per_disease)
            new_examples.extend([e for e in examples if e["animal"] == "Sheep"])
    
    return new_examples

def main():
    """Main function to augment training data"""
    print("=" * 80)
    print("TRAINING DATA AUGMENTATION FOR RARE DISEASES AND SMALL ANIMALS")
    print("=" * 80)
    
    # Load existing data
    data_file = Path("processed_data/all_processed_data.json")
    if not data_file.exists():
        print(f"❌ Error: {data_file} not found!")
        return
    
    with open(data_file, 'r') as f:
        existing_data = json.load(f)
    
    print(f"\n📊 Current Data: {len(existing_data)} samples")
    
    # Generate examples for rare diseases (0 samples)
    new_examples = []
    
    print("\n🔍 Generating examples for rare diseases:")
    for disease, info in RARE_DISEASES.items():
        # Check if disease exists in current data (more flexible matching)
        disease_count = 0
        for item in existing_data:
            item_text = (item.get("disease", "") + " " + item.get("output", "") + " " + item.get("input", "")).lower()
            if disease.lower() in item_text:
                disease_count += 1
        
        if disease_count == 0:
            print(f"  ✅ {disease}: Generating 25 examples (0 current)")
            examples = generate_training_examples(disease, info, num_examples=25)
            new_examples.extend(examples)
        elif disease in NEED_MORE:
            target = NEED_MORE[disease]["target"]
            needed = max(0, target - disease_count)
            if needed > 0:
                print(f"  ⚠️  {disease}: Current {disease_count}, generating {needed} more")
                examples = generate_training_examples(disease, info, num_examples=needed)
                new_examples.extend(examples)
        elif disease_count < 10:
            # Generate more for diseases with very few samples
            needed = 15 - disease_count
            if needed > 0:
                print(f"  ⚠️  {disease}: Current {disease_count}, generating {needed} more")
                examples = generate_training_examples(disease, info, num_examples=needed)
                new_examples.extend(examples)
    
    # Augment small animal examples
    print("\n🐐🐑 Augmenting small animal examples:")
    small_animal_examples = augment_small_animal_examples(existing_data, target_goat=100, target_sheep=100)
    new_examples.extend(small_animal_examples)
    print(f"  Generated {len(small_animal_examples)} additional small animal examples")
    
    # Combine with existing data
    augmented_data = existing_data + new_examples
    
    # Shuffle
    random.shuffle(augmented_data)
    
    print(f"\n📈 New Total: {len(augmented_data)} samples (+{len(new_examples)} new)")
    
    # Save augmented data
    output_file = Path("processed_data/all_processed_data_augmented.json")
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print(f"\n✅ Augmented data saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original samples: {len(existing_data)}")
    print(f"New samples: {len(new_examples)}")
    print(f"Total samples: {len(augmented_data)}")
    print(f"\n📁 Output file: {output_file}")
    print("\n💡 Next steps:")
    print("  1. Review the augmented data")
    print("  2. Update training script to use: processed_data/all_processed_data_augmented.json")
    print("  3. Retrain the model with augmented data")

if __name__ == "__main__":
    main()

