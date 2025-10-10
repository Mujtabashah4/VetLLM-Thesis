#!/usr/bin/env python3
"""
VetLLM Data Preprocessing Script
Handles data preparation, augmentation, and SNOMED-CT code processing
"""

import json
import os
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import requests
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SNOMEDProcessor:
    """Process SNOMED-CT codes and hierarchies"""
    
    def __init__(self, snomed_file: str = None):
        self.code_hierarchy = {}
        self.code_descriptions = {}
        self.parent_child_map = defaultdict(list)
        
        if snomed_file and os.path.exists(snomed_file):
            self.load_snomed_codes(snomed_file)
    
    def load_snomed_codes(self, file_path: str):
        """Load SNOMED-CT codes from file"""
        logger.info(f"Loading SNOMED codes from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            snomed_data = json.load(f)
        
        for code_info in snomed_data:
            code_id = code_info['code_id']
            self.code_descriptions[code_id] = code_info['description']
            self.code_hierarchy[code_id] = code_info.get('depth', 0)
            
            # Build parent-child relationships
            if 'parent_codes' in code_info:
                for parent in code_info['parent_codes']:
                    self.parent_child_map[parent].append(code_id)
        
        logger.info(f"Loaded {len(self.code_descriptions)} SNOMED codes")
    
    def create_synthetic_snomed_data(self) -> Dict:
        """Create synthetic SNOMED-CT data for demonstration"""
        synthetic_codes = {
            # Cardiovascular conditions
            "194828000": {"description": "Angina", "depth": 3, "parent_codes": ["195111005"]},
            "195111005": {"description": "Cardiovascular disease", "depth": 2, "parent_codes": ["64572001"]},
            "64572001": {"description": "Disease", "depth": 1, "parent_codes": []},
            
            # Respiratory conditions
            "195967001": {"description": "Asthma", "depth": 3, "parent_codes": ["195949008"]},
            "195949008": {"description": "Respiratory disease", "depth": 2, "parent_codes": ["64572001"]},
            
            # Infectious diseases
            "840539006": {"description": "COVID-19", "depth": 4, "parent_codes": ["87628006"]},
            "87628006": {"description": "Bacterial infectious disease", "depth": 3, "parent_codes": ["191415002"]},
            "191415002": {"description": "Infectious disease", "depth": 2, "parent_codes": ["64572001"]},
            
            # Veterinary specific codes
            "447962005": {"description": "Canine parvovirus infection", "depth": 4, "parent_codes": ["87628006"]},
            "363680008": {"description": "Feline leukemia", "depth": 4, "parent_codes": ["87628006"]},
            "14304000": {"description": "Hip dysplasia", "depth": 3, "parent_codes": ["195967001"]},
            "16973004": {"description": "Lameness", "depth": 3, "parent_codes": ["195967001"]},
            "57676002": {"description": "Joint pain", "depth": 3, "parent_codes": ["195967001"]},
            "49727002": {"description": "Cough", "depth": 3, "parent_codes": ["195949008"]},
            "397983004": {"description": "Lethargy", "depth": 3, "parent_codes": ["64572001"]},
            "79890006": {"description": "Loss of appetite", "depth": 3, "parent_codes": ["64572001"]},
            "422400008": {"description": "Vomiting", "depth": 3, "parent_codes": ["195967001"]},
            "62315008": {"description": "Diarrhea", "depth": 3, "parent_codes": ["195967001"]},
            "271807003": {"description": "Skin rash", "depth": 3, "parent_codes": ["195967001"]},
            "424492005": {"description": "Scratching", "depth": 3, "parent_codes": ["195967001"]},
            "91175000": {"description": "Seizure", "depth": 3, "parent_codes": ["64572001"]},
            "89362005": {"description": "Weight loss", "depth": 3, "parent_codes": ["64572001"]},
            "64531003": {"description": "Nasal discharge", "depth": 3, "parent_codes": ["195949008"]},
            "271860004": {"description": "Abdominal distension", "depth": 3, "parent_codes": ["195967001"]},
            "25786006": {"description": "Behavioral changes", "depth": 3, "parent_codes": ["64572001"]}
        }
        
        return synthetic_codes

class VeterinaryDataProcessor:
    """Process and augment veterinary clinical data"""
    
    def __init__(self, snomed_processor: SNOMEDProcessor):
        self.snomed_processor = snomed_processor
        self.medical_synonyms = self.load_medical_synonyms()
    
    def load_medical_synonyms(self) -> Dict[str, List[str]]:
        """Load medical term synonyms for data augmentation"""
        return {
            "dog": ["canine", "puppy", "hound", "pooch"],
            "cat": ["feline", "kitten", "kitty", "feline patient"],
            "horse": ["equine", "mare", "stallion", "gelding"],
            "cow": ["bovine", "cattle", "heifer", "bull"],
            "examination": ["exam", "checkup", "evaluation", "assessment", "inspection"],
            "temperature": ["temp", "fever", "pyrexia", "hyperthermia"],
            "breathing": ["respiration", "respiratory rate", "breathing pattern", "ventilation"],
            "heart": ["cardiac", "cardiovascular", "heart rate", "pulse"],
            "pain": ["discomfort", "soreness", "tenderness", "ache"],
            "infection": ["inflammation", "sepsis", "bacterial infection", "pathogen"],
            "weight": ["body weight", "mass", "weight status", "body condition"],
            "appetite": ["eating", "food intake", "feeding behavior", "hunger"],
            "vomiting": ["emesis", "throwing up", "regurgitation"],
            "diarrhea": ["loose stool", "watery stool", "soft feces"],
            "lethargy": ["fatigue", "tiredness", "weakness", "low energy"],
            "coughing": ["cough", "respiratory distress", "throat clearing"],
            "limping": ["lameness", "gait abnormality", "favoring leg"],
            "seizure": ["convulsion", "fit", "epileptic episode"]
        }
    
    def augment_clinical_note(self, note: str) -> List[str]:
        """Create augmented versions of clinical notes"""
        augmented_notes = [note]  # Original note
        
        # Synonym replacement (limit to 2 variations to avoid explosion)
        synonym_count = 0
        for original, synonyms in self.medical_synonyms.items():
            if original in note.lower() and synonym_count < 2:
                for synonym in synonyms[:1]:  # Use only first synonym
                    augmented_note = note.lower().replace(original, synonym)
                    augmented_notes.append(augmented_note.capitalize())
                    synonym_count += 1
                    break
        
        # Add context variations
        contexts = [
            "Emergency presentation: ",
            "Routine wellness exam: ",
            "Follow-up examination: "
        ]
        
        for context in contexts[:1]:  # Use only first context
            augmented_notes.append(context + note)
        
        return list(set(augmented_notes))  # Remove duplicates
    
    def create_instruction_variations(self) -> List[str]:
        """Create different instruction phrasings for veterinary diagnosis"""
        return [
            "Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes.",
            "Based on the clinical findings, identify the most likely SNOMED-CT diagnosis codes.",
            "What are the appropriate SNOMED-CT codes for this veterinary case?",
            "Determine the diagnosis codes that best match this clinical presentation.",
            "Predict the SNOMED-CT classifications for this veterinary patient.",
            "Identify the diagnostic codes based on the clinical notes provided.",
            "What diagnoses would you assign to this veterinary case using SNOMED-CT codes?"
        ]
    
    def process_veterinary_dataset(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw veterinary data into instruction format with augmentation"""
        processed_data = []
        instruction_variations = self.create_instruction_variations()
        
        logger.info(f"Processing {len(raw_data)} veterinary samples")
        
        for i, item in enumerate(raw_data):
            clinical_note = item['clinical_note']
            snomed_codes = item['snomed_codes']
            
            # Basic processing
            base_sample = {
                "instruction": random.choice(instruction_variations),
                "input": f"Clinical Note: {clinical_note}",
                "output": f"Diagnosed conditions: {', '.join(snomed_codes)}",
                "snomed_codes": snomed_codes
            }
            processed_data.append(base_sample)
            
            # Create augmented versions (limited to avoid excessive data)
            augmented_notes = self.augment_clinical_note(clinical_note)
            
            for aug_note in augmented_notes[1:2]:  # Use only 1 augmented version
                aug_sample = {
                    "instruction": random.choice(instruction_variations),
                    "input": f"Clinical Note: {aug_note}",
                    "output": f"Diagnosed conditions: {', '.join(snomed_codes)}",
                    "snomed_codes": snomed_codes
                }
                processed_data.append(aug_sample)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(raw_data)} samples")
        
        logger.info(f"Generated {len(processed_data)} processed samples")
        return processed_data

class AlpacaDataProcessor:
    """Process Stanford Alpaca dataset"""
    
    def __init__(self, alpaca_file: str):
        self.alpaca_file = alpaca_file
    
    def download_alpaca_dataset(self, output_path: str) -> bool:
        """Download Stanford Alpaca dataset"""
        alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        
        logger.info(f"Downloading Alpaca dataset from {alpaca_url}")
        
        try:
            response = requests.get(alpaca_url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2)
            
            logger.info(f"Downloaded Alpaca dataset to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Alpaca dataset: {e}")
            return False
    
    def load_alpaca_dataset(self) -> List[Dict]:
        """Load Alpaca dataset"""
        if not os.path.exists(self.alpaca_file):
            logger.info("Alpaca dataset not found. Attempting to download...")
            if not self.download_alpaca_dataset(self.alpaca_file):
                raise FileNotFoundError(f"Could not load or download Alpaca dataset: {self.alpaca_file}")
        
        with open(self.alpaca_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} Alpaca samples")
        return data
    
    def filter_alpaca_for_medical(self, data: List[Dict], keywords: Set[str]) -> List[Dict]:
        """Filter Alpaca dataset for medical-related instructions"""
        medical_data = []
        
        for item in data:
            instruction = item['instruction'].lower()
            input_text = item.get('input', '').lower()
            
            # Check if any medical keywords are present
            if any(keyword in instruction + ' ' + input_text for keyword in keywords):
                medical_data.append(item)
        
        logger.info(f"Filtered to {len(medical_data)} medical-related samples")
        return medical_data

def create_synthetic_veterinary_data(num_samples: int = 1000) -> List[Dict]:
    """Create synthetic veterinary clinical data for demonstration"""
    
    logger.info(f"Creating {num_samples} synthetic veterinary samples")
    
    # Sample clinical notes templates
    note_templates = [
        "Patient: {species} {breed}, {age} years old, {weight}kg. Chief complaint: {complaint}. Physical examination reveals {findings}. Temperature: {temp}°C, Heart rate: {hr} bpm.",
        "{species} presenting with {complaint}. On examination: {findings}. Vital signs: T={temp}°C, HR={hr}bpm, RR={rr}bpm. Assessment: {assessment}.",
        "{age}-year-old {breed} {species} with history of {complaint}. Physical findings: {findings}. Diagnostic impression: {assessment}.",
        "Emergency presentation: {species} with acute {complaint}. Clinical signs: {findings}. Vital parameters: Temperature {temp}°C, pulse {hr}bpm.",
        "Routine examination: {species}, {breed}, {age} years. Owner reports {complaint}. Examination findings: {findings}."
    ]
    
    # Sample data for templates
    species_options = ["dog", "cat", "rabbit", "bird", "horse", "cow"]
    breed_options = {
        "dog": ["Labrador", "German Shepherd", "Golden Retriever", "Bulldog", "Beagle", "Poodle", "Husky"],
        "cat": ["Persian", "Siamese", "Maine Coon", "British Shorthair", "Ragdoll", "Domestic Shorthair"],
        "rabbit": ["Holland Lop", "Netherland Dwarf", "Mini Rex", "Lionhead", "Angora"],
        "bird": ["Cockatiel", "Budgie", "Canary", "Parrot", "Finch", "Lovebird"],
        "horse": ["Thoroughbred", "Quarter Horse", "Arabian", "Pinto", "Clydesdale"],
        "cow": ["Holstein", "Angus", "Hereford", "Jersey", "Brahman"]
    }
    
    complaints = [
        "lethargy and decreased appetite",
        "vomiting and diarrhea",
        "difficulty breathing",
        "limping and joint pain",
        "skin irritation and scratching",
        "seizure activity",
        "weight loss",
        "coughing and nasal discharge",
        "abdominal distension",
        "behavioral changes",
        "fever and dehydration",
        "loss of balance",
        "excessive thirst",
        "difficulty urinating",
        "eye discharge"
    ]
    
    findings = [
        "mild dehydration, pale mucous membranes",
        "elevated temperature, tachycardia",
        "respiratory distress, abnormal lung sounds",
        "joint swelling, pain on palpation",
        "skin lesions, erythema",
        "neurological abnormalities",
        "muscle wasting, poor body condition",
        "nasal discharge, enlarged lymph nodes",
        "fluid accumulation in abdomen",
        "aggressive behavior, disorientation",
        "abnormal heart rhythm",
        "enlarged liver on palpation",
        "dental disease, halitosis",
        "corneal opacity",
        "decreased reflexes"
    ]
    
    snomed_mappings = {
        "lethargy and decreased appetite": ["397983004", "79890006"],
        "vomiting and diarrhea": ["422400008", "62315008"],
        "difficulty breathing": ["267036007"],
        "limping and joint pain": ["16973004", "57676002"],
        "skin irritation and scratching": ["271807003", "424492005"],
        "seizure activity": ["91175000"],
        "weight loss": ["89362005"],
        "coughing and nasal discharge": ["49727002", "64531003"],
        "abdominal distension": ["271860004"],
        "behavioral changes": ["25786006"],
        "fever and dehydration": ["386661006", "34095006"],
        "loss of balance": ["387603000"],
        "excessive thirst": ["17173007"],
        "difficulty urinating": ["139394000"],
        "eye discharge": ["246636008"]
    }
    
    synthetic_data = []
    
    for i in range(num_samples):
        species = random.choice(species_options)
        breed = random.choice(breed_options.get(species, ["Mixed"]))
        age = random.randint(1, 15)
        weight = round(random.uniform(0.5, 50.0), 1)
        complaint = random.choice(complaints)
        finding = random.choice(findings)
        temp = round(random.uniform(37.5, 41.0), 1)
        hr = random.randint(60, 180)
        rr = random.randint(10, 40)
        
        template = random.choice(note_templates)
        clinical_note = template.format(
            species=species,
            breed=breed,
            age=age,
            weight=weight,
            complaint=complaint,
            findings=finding,
            assessment=complaint,
            temp=temp,
            hr=hr,
            rr=rr
        )
        
        # Get SNOMED codes for the complaint
        codes = snomed_mappings.get(complaint, ["64572001"])  # Default to "Disease"
        
        synthetic_data.append({
            "clinical_note": clinical_note,
            "snomed_codes": codes,
            "species": species,
            "age": age,
            "complaint": complaint
        })
        
        if (i + 1) % 200 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} synthetic samples")
    
    logger.info(f"Created {len(synthetic_data)} synthetic veterinary samples")
    return synthetic_data

def main():
    """Main data preprocessing pipeline"""
    parser = argparse.ArgumentParser(description="VetLLM Data Preprocessing")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--num-synthetic", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--download-alpaca", action="store_true", help="Download Alpaca dataset")
    parser.add_argument("--augment", action="store_true", default=True, help="Apply data augmentation")
    
    args = parser.parse_args()
    
    # Setup directories
    data_dir = Path(args.output_dir)
    data_dir.mkdir(exist_ok=True)
    
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    logger.info("Starting VetLLM data preprocessing pipeline")
    logger.info(f"Output directory: {data_dir}")
    
    # Initialize processors
    snomed_processor = SNOMEDProcessor()
    
    # Create synthetic SNOMED data if not available
    synthetic_snomed = snomed_processor.create_synthetic_snomed_data()
    snomed_file = data_dir / "snomed_codes.json"
    
    with open(snomed_file, 'w') as f:
        json.dump([{
            "code_id": code_id,
            "description": info["description"],
            "depth": info["depth"],
            "parent_codes": info["parent_codes"]
        } for code_id, info in synthetic_snomed.items()], f, indent=2)
    
    logger.info(f"Created synthetic SNOMED-CT codes data: {snomed_file}")
    
    # Load SNOMED processor with synthetic data
    snomed_processor.load_snomed_codes(snomed_file)
    
    # Initialize data processors
    vet_processor = VeterinaryDataProcessor(snomed_processor)
    alpaca_processor = AlpacaDataProcessor(str(data_dir / "alpaca_data.json"))
    
    # Process Alpaca dataset
    alpaca_data = []
    if args.download_alpaca:
        try:
            alpaca_data = alpaca_processor.load_alpaca_dataset()
            logger.info(f"Loaded {len(alpaca_data)} Alpaca samples")
            
            # Filter for medical-related content
            medical_keywords = {
                'medical', 'health', 'disease', 'symptom', 'treatment', 'diagnosis',
                'patient', 'clinical', 'therapy', 'medicine', 'drug', 'hospital',
                'veterinary', 'animal', 'pet', 'dog', 'cat', 'horse'
            }
            
            medical_alpaca = alpaca_processor.filter_alpaca_for_medical(alpaca_data, medical_keywords)
            
            # Save filtered data
            with open(processed_dir / "alpaca_medical_filtered.json", 'w') as f:
                json.dump(medical_alpaca, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not process Alpaca dataset: {e}")
            medical_alpaca = []
    else:
        medical_alpaca = []
    
    # Create synthetic veterinary data
    logger.info("Creating synthetic veterinary data...")
    synthetic_vet_data = create_synthetic_veterinary_data(num_samples=args.num_synthetic)
    
    # Process veterinary data
    processed_vet_data = vet_processor.process_veterinary_dataset(synthetic_vet_data)
    
    # Save processed veterinary data
    with open(data_dir / "veterinary_notes.json", 'w') as f:
        json.dump(synthetic_vet_data, f, indent=2)
    
    with open(processed_dir / "veterinary_processed.json", 'w') as f:
        json.dump(processed_vet_data, f, indent=2)
    
    # Create combined training dataset
    combined_data = []
    
    # Add processed veterinary data
    combined_data.extend(processed_vet_data)
    
    # Add Alpaca data (if available)
    if medical_alpaca:
        # Take a subset of Alpaca data to balance with veterinary data
        alpaca_subset = random.sample(medical_alpaca, min(len(medical_alpaca), len(processed_vet_data) // 2))
        combined_data.extend(alpaca_subset)
    
    # Shuffle combined data
    random.shuffle(combined_data)
    
    # Split into train/validation/test
    train_split = int(0.8 * len(combined_data))
    val_split = int(0.9 * len(combined_data))
    
    train_data = combined_data[:train_split]
    val_data = combined_data[train_split:val_split]
    test_data = combined_data[val_split:]
    
    # Save splits
    with open(processed_dir / "train_data.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(processed_dir / "val_data.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(processed_dir / "test_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Generate data statistics
    stats = {
        "total_samples": len(combined_data),
        "veterinary_samples": len(processed_vet_data),
        "alpaca_samples": len(medical_alpaca) if medical_alpaca else 0,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "snomed_codes": len(synthetic_snomed),
        "processing_date": datetime.now().isoformat(),
        "parameters": {
            "num_synthetic": args.num_synthetic,
            "augmentation_enabled": args.augment,
            "alpaca_downloaded": args.download_alpaca
        }
    }
    
    with open(processed_dir / "data_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "="*50)
    logger.info("DATA PREPROCESSING COMPLETED!")
    logger.info("="*50)
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    logger.info(f"Total SNOMED codes: {len(synthetic_snomed)}")
    logger.info(f"Data statistics saved to: {processed_dir / 'data_stats.json'}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
