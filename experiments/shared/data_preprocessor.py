"""
Unified Data Preprocessor for VetLLM Experiments
Supports Llama 3.1 8B and Qwen2.5 7B chat formats

Author: VetLLM Research Team
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
import hashlib
from collections import Counter
import random

# ============================================
# Configuration
# ============================================

@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    max_length: int = 512
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    remove_duplicates: bool = True  # Default: deduplicate to avoid data leakage
    max_duplicates_per_case: int = 0  # 0 = no limit, >0 = max copies per unique case
    balance_classes: bool = False  # Optional: balance disease classes
    min_samples_per_class: int = 5


# ============================================
# SNOMED-CT Mapping
# ============================================

DISEASE_TO_SNOMED = {
    # Cow/Buffalo Diseases (with actual SNOMED-CT codes where available)
    "Anaplasmosis": ["15264008"],
    "Anthrax": ["40214000"],
    "Babesiosis": ["24026003"],
    "BABESIOSIS": ["24026003"],
    "Black Quarter": ["29600000"],
    "B.Q": ["29600000"],
    "Brucellosis": ["75702008"],
    "CCP": ["2260006"],
    "CCPP": ["2260006"],
    "Fascioliasis": ["4764006"],
    "Liver Fluke": ["4764006"],
    "FMD": ["3974005"],
    "Foot and Mouth": ["3974005"],
    "Foot and mouth": ["3974005"],
    "HC": ["74942003"],
    "H.S": ["198462004"],
    "Interotoximia": ["370514003"],
    "Mastitis": ["72934000"],
    "Mastits": ["72934000"],
    "Metritis": ["50868007"],
    "Pox": ["363196005"],
    "PPR": ["1679004"],
    "P.P.R": ["1679004"],
    "Kataa": ["1679004"],
    "Goat Pox": ["57428005"],
    "Rabies": ["14146002"],
    "Rabbies": ["14146002"],
    "Theileriosis": ["24694002"],
    "Tuberculosis": ["56717001"],
    "T.B": ["56717001"],
    # Additional diseases without standard codes (using placeholder)
    "Abortion": ["VET_ABORT_001"],
    "Flue": ["VET_FLUE_001"],
    "Foot Rot": ["VET_FOOTROT_001"],
    "Fracture of the Leg": ["VET_FRACTURE_001"],
    "Infection": ["VET_INFECT_001"],
    "Internal Worms": ["VET_WORMS_001"],
    "Ketosis": ["VET_KETOSIS_001"],
    "Lamititus": ["VET_LAMINITIS_001"],
    "laminitis": ["VET_LAMINITIS_001"],
    "Tympany": ["VET_TYMPANY_001"],
}

# Disease name normalization mapping
DISEASE_NORMALIZATION = {
    "B.Q": "Black Quarter",
    "BABESIOSIS": "Babesiosis",
    "Mastits": "Mastitis",
    "Rabbies": "Rabies",
    "T.B": "Tuberculosis",
    "H.S": "Hemorrhagic Septicemia",
    "FMD": "Foot and Mouth Disease",
    "Foot and mouth": "Foot and Mouth Disease",
    "Foot and Mouth": "Foot and Mouth Disease",
    "PPR": "Peste des Petits Ruminants",
    "P.P.R": "Peste des Petits Ruminants",
    "Kataa": "Peste des Petits Ruminants",
    "CCPP": "Contagious Caprine Pleuropneumonia",
    "CCP": "Contagious Caprine Pleuropneumonia",
    "HC": "Hemorrhagic Fever",
    "Lamititus": "Laminitis",
    "laminitis": "Laminitis",
}


# ============================================
# Symptom Formatting
# ============================================

SYMPTOM_MAPPINGS = {
    'fever(f)': 'fever',
    'very high fever': 'high fever',
    'continuous loose motions': 'persistent diarrhea',
    'loose motions': 'diarrhea',
    'loose motions with blood': 'bloody diarrhea (dysentery)',
    'blood leakage from nose': 'epistaxis (nasal bleeding)',
    'difficulty in breathing': 'dyspnea (respiratory distress)',
    'breathing while mouth open': 'open-mouth breathing',
    'rapid breathing': 'tachypnea',
    'mouth frothing': 'oral frothing (salivation)',
    'drop in milk production': 'decreased milk production',
    'shortage of milk': 'reduced milk yield',
    'milk in semi solid form': 'abnormal milk consistency (clotted)',
    'blisters in mouth': 'oral vesicles/ulcers',
    'blisters on lips': 'labial vesicles',
    'blisters on feet': 'foot vesicles/lesions',
    'neck swelling': 'cervical edema',
    'tongue coming out of mouth': 'tongue protrusion',
    'teat gnarls': 'teat lesions/nodules',
    'gnarls in teats': 'teat lesions/nodules',
    'vaginal gnarls': 'vaginal lesions',
    'air/ crapitatifeelings at surface of any body part': 'subcutaneous emphysema (crepitus)',
    'color change of muscles of limb': 'limb muscle discoloration',
    'sudden death': 'sudden death (peracute)',
    'water or fluid leakage from eyes,nose or mouth': 'mucopurulent discharge',
    'fluid leakage from nose': 'nasal discharge',
    'sever cough': 'severe coughing',
    'pain in stomach and screams': 'severe abdominal pain (colic)',
    'blood in milk': 'bloody milk (mastitis indicator)',
    'udder become swollen': 'udder edema/mastitis',
    'lesion on un_hairy parts of body': 'skin lesions on glabrous areas',
    'swollen under jaw': 'submandibular edema (bottle jaw)',
    'hair fall': 'alopecia',
    'stiffening of body': 'muscle rigidity/tetany',
    'pain and anxiety': 'restlessness and discomfort',
    'mites': 'mange/mite infestation',
    'weakness': 'generalized weakness',
    'cough': 'coughing',
    'abortion': 'abortion/pregnancy loss',
}


def format_symptom(symptom: str) -> str:
    """Format a symptom name to be more clinically descriptive."""
    symptom_lower = symptom.lower().strip()
    return SYMPTOM_MAPPINGS.get(symptom_lower, symptom_lower)


def normalize_disease(disease: str) -> str:
    """Normalize disease name to standard form."""
    disease_clean = str(disease).strip()
    return DISEASE_NORMALIZATION.get(disease_clean, disease_clean)


# ============================================
# Chat Format Templates
# ============================================

class ChatFormatTemplate:
    """Base class for model-specific chat formats."""
    
    @staticmethod
    def get_system_prompt() -> str:
        return """You are VetLLM, a veterinary clinical assistant specialized in livestock diseases. 
Given a clinical case presentation, you will:
1. Analyze the clinical signs and symptoms
2. Provide the most likely diagnosis with SNOMED-CT codes when available
3. List differential diagnoses to consider
4. Recommend appropriate treatment and management steps
5. Explain your clinical reasoning"""

    def format_input(self, case: Dict) -> str:
        raise NotImplementedError
    
    def format_output(self, case: Dict) -> str:
        raise NotImplementedError
    
    def format_conversation(self, case: Dict) -> str:
        raise NotImplementedError


class Llama31ChatFormat(ChatFormatTemplate):
    """Llama 3.1 8B Instruct chat format."""
    
    def format_input(self, case: Dict) -> str:
        """Format the user input for Llama 3.1."""
        species = case.get('animal', case.get('species', 'Unknown'))
        symptoms = case.get('symptoms', [])
        
        if isinstance(symptoms, list):
            formatted_symptoms = [format_symptom(s) for s in symptoms]
            if len(formatted_symptoms) == 1:
                symptoms_text = formatted_symptoms[0]
            elif len(formatted_symptoms) == 2:
                symptoms_text = f"{formatted_symptoms[0]} and {formatted_symptoms[1]}"
            else:
                symptoms_text = ", ".join(formatted_symptoms[:-1]) + f", and {formatted_symptoms[-1]}"
        else:
            symptoms_text = str(symptoms)
        
        return f"""Species: {species}
Clinical presentation: {symptoms_text}
Physical examination findings: As described above

Please provide:
1. Most likely diagnosis (with SNOMED-CT code if available)
2. Differential diagnoses
3. Recommended treatment plan
4. Clinical reasoning"""

    def format_output(self, case: Dict) -> str:
        """Format the expected output for Llama 3.1."""
        disease = case.get('disease', 'Unknown')
        normalized_disease = normalize_disease(disease)
        snomed_codes = case.get('snomed_codes', [])
        symptoms = case.get('symptoms', [])
        
        # Build SNOMED code string
        if snomed_codes:
            snomed_str = ", ".join(snomed_codes)
            diagnosis_line = f"**{normalized_disease}** (SNOMED-CT: {snomed_str})"
        else:
            diagnosis_line = f"**{normalized_disease}**"
        
        # Generate differentials based on species and symptoms
        differentials = self._generate_differentials(case)
        diff_text = "\n".join([f"   - {d}" for d in differentials])
        
        # Generate treatment based on disease
        treatment = self._generate_treatment(normalized_disease, case.get('animal', 'Unknown'))
        
        return f"""1. **Primary Diagnosis**: {diagnosis_line}

2. **Differential Diagnoses**:
{diff_text}

3. **Recommended Treatment**:
{treatment}

4. **Clinical Reasoning**:
   Based on the presenting clinical signs ({', '.join(symptoms[:3])}{'...' if len(symptoms) > 3 else ''}), the pattern is consistent with {normalized_disease}. The combination of symptoms and species-specific epidemiology supports this diagnosis."""

    def _generate_differentials(self, case: Dict) -> List[str]:
        """Generate plausible differential diagnoses."""
        disease = normalize_disease(case.get('disease', ''))
        species = case.get('animal', '').lower()
        
        # Disease-specific differentials
        differentials_map = {
            "Foot and Mouth Disease": ["Vesicular Stomatitis", "Bluetongue", "Bovine Viral Diarrhea"],
            "Anthrax": ["Blackleg", "Hemorrhagic Septicemia", "Clostridial diseases"],
            "Black Quarter": ["Anthrax", "Malignant Edema", "Bacillary Hemoglobinuria"],
            "Mastitis": ["Udder edema", "Teat injury", "Mammary abscess"],
            "Peste des Petits Ruminants": ["Goat Pox", "Contagious Ecthyma", "Bluetongue"],
            "Brucellosis": ["Trichomoniasis", "Campylobacteriosis", "Leptospirosis"],
            "Babesiosis": ["Anaplasmosis", "Theileriosis", "Trypanosomiasis"],
        }
        
        base_differentials = differentials_map.get(disease, ["Infectious disease", "Metabolic disorder", "Toxicosis"])
        return [d for d in base_differentials if d != disease][:3]

    def _generate_treatment(self, disease: str, species: str) -> str:
        """Generate treatment recommendations."""
        treatments = {
            "Foot and Mouth Disease": "Supportive care, isolation, wound care for lesions. Notify authorities (notifiable disease).",
            "Anthrax": "High-dose penicillin, oxytetracycline. Carcass disposal protocols. Vaccination of contacts.",
            "Black Quarter": "High-dose penicillin G, anti-inflammatory therapy. Surgical debridement if localized.",
            "Mastitis": "Intramammary antibiotics, systemic antibiotics if systemic signs. Frequent milking.",
            "Peste des Petits Ruminants": "Supportive care, antibiotics for secondary infections. Vaccination of herd.",
            "Babesiosis": "Diminazene aceturate or imidocarb dipropionate. Supportive fluid therapy.",
            "Brucellosis": "Test and cull policy (no treatment in animals). Notify authorities.",
        }
        return treatments.get(disease, "Symptomatic treatment, supportive care, and appropriate antimicrobials based on clinical presentation.")

    def format_conversation(self, case: Dict) -> str:
        """Format the full conversation in Llama 3.1 format."""
        system = self.get_system_prompt()
        user = self.format_input(case)
        assistant = self.format_output(case)
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>"""


class Qwen25ChatFormat(ChatFormatTemplate):
    """Qwen2.5 7B Instruct chat format (ChatML)."""
    
    def format_input(self, case: Dict) -> str:
        """Format the user input for Qwen2.5."""
        species = case.get('animal', case.get('species', 'Unknown'))
        symptoms = case.get('symptoms', [])
        
        if isinstance(symptoms, list):
            formatted_symptoms = [format_symptom(s) for s in symptoms]
            if len(formatted_symptoms) == 1:
                symptoms_text = formatted_symptoms[0]
            elif len(formatted_symptoms) == 2:
                symptoms_text = f"{formatted_symptoms[0]} and {formatted_symptoms[1]}"
            else:
                symptoms_text = ", ".join(formatted_symptoms[:-1]) + f", and {formatted_symptoms[-1]}"
        else:
            symptoms_text = str(symptoms)
        
        return f"""Analyze this veterinary case:

**Species**: {species}
**Clinical Signs**: {symptoms_text}

Provide your diagnosis, differentials, treatment plan, and reasoning."""

    def format_output(self, case: Dict) -> str:
        """Format the expected output for Qwen2.5."""
        # Reuse Llama format for consistency in outputs
        llama_format = Llama31ChatFormat()
        return llama_format.format_output(case)

    def format_conversation(self, case: Dict) -> str:
        """Format the full conversation in Qwen2.5 ChatML format."""
        system = self.get_system_prompt()
        user = self.format_input(case)
        assistant = self.format_output(case)
        
        return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""


# ============================================
# Data Processing Functions
# ============================================

def extract_symptoms_from_row(row: pd.Series, symptom_columns: List[str]) -> List[str]:
    """Extract symptoms that are present (value = 1) from a row."""
    symptoms = []
    for col in symptom_columns:
        if col in row.index:
            val = row[col]
            if val == 1 or val == True or (pd.notna(val) and str(val).strip() == '1'):
                symptoms.append(col)
    return symptoms


def process_excel_to_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Process an Excel file and extract case data.
    
    Args:
        file_path: Path to the Excel file
    
    Returns:
        List of case dictionaries
    """
    print(f"Processing: {file_path}")
    
    df = pd.read_excel(file_path)
    
    # Identify columns
    animal_col = 'Animal Name'
    disease_col = 'Disease'
    
    # Get symptom columns
    symptom_columns = [col for col in df.columns 
                      if col not in [animal_col, disease_col] 
                      and not str(col).startswith('Unnamed')]
    
    print(f"  Found {len(df)} rows, {len(symptom_columns)} symptom columns")
    
    cases = []
    for idx, row in df.iterrows():
        try:
            animal = str(row[animal_col]).strip() if pd.notna(row[animal_col]) else "Unknown"
            disease = row[disease_col] if pd.notna(row[disease_col]) else None
            
            if not disease:
                continue
            
            symptoms = extract_symptoms_from_row(row, symptom_columns)
            if not symptoms:
                continue
            
            disease_str = str(disease).strip()
            snomed_codes = DISEASE_TO_SNOMED.get(disease_str, [])
            
            case = {
                'animal': animal,
                'disease': disease_str,
                'disease_normalized': normalize_disease(disease_str),
                'symptoms': symptoms,
                'snomed_codes': snomed_codes,
                'source_file': os.path.basename(file_path),
                'source_row': idx + 2  # Account for 0-index and header
            }
            cases.append(case)
            
        except Exception as e:
            print(f"  Warning: Error processing row {idx + 2}: {e}")
            continue
    
    print(f"  Extracted {len(cases)} valid cases")
    return cases


def deduplicate_cases(cases: List[Dict]) -> List[Dict]:
    """Remove duplicate cases based on content hash."""
    seen_hashes = set()
    unique_cases = []
    
    for case in cases:
        # Create hash from key fields
        content = f"{case['animal']}|{case['disease']}|{sorted(case['symptoms'])}"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            case['content_hash'] = content_hash
            unique_cases.append(case)
    
    print(f"Deduplication: {len(cases)} -> {len(unique_cases)} cases")
    return unique_cases


def create_train_val_test_split(
    cases: List[Dict],
    config: PreprocessConfig
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test sets with stratification by disease.
    
    Uses case-level splitting to avoid data leakage.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Get disease labels for stratification
    diseases = [case['disease_normalized'] for case in cases]
    disease_counts = Counter(diseases)
    
    # Filter out classes with too few samples for stratification
    min_samples = 2  # Need at least 2 for stratified split
    valid_diseases = {d for d, c in disease_counts.items() if c >= min_samples}
    
    # Assign numeric labels, grouping rare diseases
    disease_to_label = {}
    label_counter = 0
    for disease in sorted(set(diseases)):
        if disease in valid_diseases:
            disease_to_label[disease] = label_counter
            label_counter += 1
        else:
            disease_to_label[disease] = -1  # Rare class
    
    labels = [disease_to_label[case['disease_normalized']] for case in cases]
    
    # First split: train vs (val + test)
    val_test_ratio = config.val_ratio + config.test_ratio
    
    try:
        train_cases, val_test_cases, train_labels, val_test_labels = train_test_split(
            cases, labels,
            test_size=val_test_ratio,
            random_state=config.seed,
            stratify=labels
        )
    except ValueError:
        # Fall back to non-stratified if stratification fails
        print("Warning: Stratified split failed, using random split")
        train_cases, val_test_cases = train_test_split(
            cases,
            test_size=val_test_ratio,
            random_state=config.seed
        )
        val_test_labels = [disease_to_label[c['disease_normalized']] for c in val_test_cases]
    
    # Second split: val vs test
    relative_test_ratio = config.test_ratio / val_test_ratio
    
    try:
        val_cases, test_cases = train_test_split(
            val_test_cases,
            test_size=relative_test_ratio,
            random_state=config.seed,
            stratify=val_test_labels
        )
    except ValueError:
        val_cases, test_cases = train_test_split(
            val_test_cases,
            test_size=relative_test_ratio,
            random_state=config.seed
        )
    
    print(f"Split sizes: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}")
    
    return train_cases, val_cases, test_cases


def format_cases_for_model(
    cases: List[Dict],
    model_type: str = "llama3.1"
) -> List[Dict]:
    """
    Format cases for a specific model's chat format.
    
    Args:
        cases: List of case dictionaries
        model_type: "llama3.1" or "qwen2.5"
    
    Returns:
        List of formatted training samples
    """
    if model_type == "llama3.1":
        formatter = Llama31ChatFormat()
    elif model_type == "qwen2.5":
        formatter = Qwen25ChatFormat()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    formatted = []
    for case in cases:
        formatted_case = {
            'text': formatter.format_conversation(case),
            'input': formatter.format_input(case),
            'output': formatter.format_output(case),
            'metadata': {
                'animal': case['animal'],
                'disease': case['disease'],
                'disease_normalized': case['disease_normalized'],
                'snomed_codes': case['snomed_codes'],
                'symptoms': case['symptoms'],
            }
        }
        formatted.append(formatted_case)
    
    return formatted


def generate_dataset_statistics(
    train: List[Dict],
    val: List[Dict], 
    test: List[Dict]
) -> Dict:
    """Generate comprehensive dataset statistics."""
    
    def get_stats(cases: List[Dict]) -> Dict:
        diseases = [c['metadata']['disease_normalized'] for c in cases]
        animals = [c['metadata']['animal'] for c in cases]
        num_symptoms = [len(c['metadata']['symptoms']) for c in cases]
        
        return {
            'num_samples': len(cases),
            'disease_distribution': dict(Counter(diseases)),
            'animal_distribution': dict(Counter(animals)),
            'avg_symptoms': np.mean(num_symptoms) if num_symptoms else 0,
            'min_symptoms': min(num_symptoms) if num_symptoms else 0,
            'max_symptoms': max(num_symptoms) if num_symptoms else 0,
        }
    
    return {
        'train': get_stats(train),
        'validation': get_stats(val),
        'test': get_stats(test),
        'total_samples': len(train) + len(val) + len(test),
    }


# ============================================
# Main Preprocessing Pipeline
# ============================================

def preprocess_dataset(
    dataset_dir: str,
    output_dir: str,
    model_type: str = "llama3.1",
    config: Optional[PreprocessConfig] = None
) -> Dict:
    """
    Main preprocessing pipeline.
    
    Args:
        dataset_dir: Path to Dataset_UVAS directory
        output_dir: Path to output directory (model-specific)
        model_type: "llama3.1" or "qwen2.5"
        config: Preprocessing configuration
    
    Returns:
        Dictionary with statistics and paths
    """
    if config is None:
        config = PreprocessConfig()
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VetLLM Data Preprocessing Pipeline")
    print(f"Model: {model_type}")
    print(f"{'='*60}\n")
    
    # Step 1: Load all Excel files
    excel_files = [
        "Verified DLO data - (Cow Buffalo).xlsx",
        "Verified DLO data (Sheep Goat).xlsx"
    ]
    
    all_cases = []
    for excel_file in excel_files:
        file_path = dataset_path / excel_file
        if file_path.exists():
            cases = process_excel_to_cases(str(file_path))
            all_cases.extend(cases)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print(f"\nTotal raw cases: {len(all_cases)}")
    
    # Step 2: Deduplicate
    if config.remove_duplicates:
        all_cases = deduplicate_cases(all_cases)
    
    # Step 3: Train/Val/Test split
    train_cases, val_cases, test_cases = create_train_val_test_split(all_cases, config)
    
    # Step 4: Format for model
    print(f"\nFormatting data for {model_type}...")
    train_formatted = format_cases_for_model(train_cases, model_type)
    val_formatted = format_cases_for_model(val_cases, model_type)
    test_formatted = format_cases_for_model(test_cases, model_type)
    
    # Step 5: Save datasets
    train_path = output_path / "train.json"
    val_path = output_path / "validation.json"
    test_path = output_path / "test.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_formatted, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_formatted, f, indent=2, ensure_ascii=False)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_formatted, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved datasets:")
    print(f"  Train: {train_path} ({len(train_formatted)} samples)")
    print(f"  Val: {val_path} ({len(val_formatted)} samples)")
    print(f"  Test: {test_path} ({len(test_formatted)} samples)")
    
    # Step 6: Generate and save statistics
    stats = generate_dataset_statistics(train_formatted, val_formatted, test_formatted)
    stats['config'] = {
        'model_type': model_type,
        'max_length': config.max_length,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'test_ratio': config.test_ratio,
        'seed': config.seed,
    }
    
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Stats: {stats_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Dataset Statistics Summary")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Train: {stats['train']['num_samples']} samples")
    print(f"  - Diseases: {len(stats['train']['disease_distribution'])} unique")
    print(f"  - Animals: {stats['train']['animal_distribution']}")
    print(f"  - Avg symptoms/case: {stats['train']['avg_symptoms']:.1f}")
    print(f"Validation: {stats['validation']['num_samples']} samples")
    print(f"Test: {stats['test']['num_samples']} samples")
    
    return {
        'train_path': str(train_path),
        'val_path': str(val_path),
        'test_path': str(test_path),
        'stats_path': str(stats_path),
        'statistics': stats,
    }


# ============================================
# CLI Interface
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VetLLM Data Preprocessor")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../../Dataset_UVAS",
        help="Path to Dataset_UVAS directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama3.1", "qwen2.5"],
        required=True,
        help="Target model type"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Keep duplicate cases (NOT recommended - causes data leakage)"
    )
    
    args = parser.parse_args()
    
    config = PreprocessConfig(
        max_length=args.max_length,
        seed=args.seed,
        remove_duplicates=not args.no_deduplicate,  # Deduplicate by default
    )
    
    preprocess_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        config=config,
    )

