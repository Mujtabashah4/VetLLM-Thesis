"""
Data Preprocessing Script for VetLLM Fine-tuning
Converts Excel data to JSON format required for fine-tuning
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any
from pathlib import Path

# ============================================
# SNOMED-CT Code Mapping
# ============================================
# Default mappings (used if snomed_mapping.json is not found)

_DEFAULT_DISEASE_TO_SNOMED = {
    # Cow/Buffalo Diseases
    'Anthrax': ['100000000000001'],
    'B.Q': ['100000000000002'],
    'BABESIOSIS': ['100000000000003'],
    'Babesiosis': ['100000000000003'],
    'Brucellosis': ['100000000000004'],
    'FMD': ['100000000000005'],
    'H.S': ['100000000000006'],
    'Mastits': ['100000000000007'],
    'Rabbies': ['100000000000008'],
    'T.B': ['100000000000009'],
    'Theileriosis': ['100000000000010'],
    # Sheep/Goat Diseases
    'Abortion': ['100000000000011'],
    'CCPP': ['100000000000012'],
    'Flue': ['100000000000013'],
    'Foot Rot': ['100000000000014'],
    'Foot and Mouth': ['100000000000005'],
    'Foot and mouth': ['100000000000005'],
    'Fracture of the Leg': ['100000000000015'],
    'Goat Pox': ['100000000000016'],
    'Infection': ['100000000000017'],
    'Internal Worms': ['100000000000018'],
    'Kataa': ['100000000000019'],
    'Ketosis': ['100000000000020'],
    'Lamititus': ['100000000000021'],
    'laminitis': ['100000000000021'],
    'Liver Fluke': ['100000000000022'],
    'Mastitis': ['100000000000007'],
    'P.P.R': ['100000000000023'],
    'PPR': ['100000000000023'],
    'Tympany': ['100000000000024'],
}

_DEFAULT_SYMPTOM_TO_SNOMED = {
    'Lameness': '16973004',
    'Mouth Frothing': '100000000000100',
    'Drop In Milk Production': '100000000000101',
    'Blisters In Mouth': '100000000000102',
    'Blisters On Feet': '100000000000103',
    'Milk In Semi Solid Form': '100000000000104',
    'Neck Swelling': '100000000000105',
    'Tongue Coming Out Of Mouth': '100000000000106',
    'Difficulty In Breathing': '267036007',
    'Breathing While Mouth Open': '267036007',
    'Continuous Loose Motions': '62315008',
    'Blood Leakage From Nose': '100000000000107',
    'Teat Gnarls': '100000000000108',
    'Air/ crapitatifeelings at surface of any Body part': '100000000000109',
    'Fever': '386661006',
    'Very High Fever': '386661006',
    'Color Change Of Muscles of limb': '100000000000110',
    'Sudden Death': '100000000000111',
    'Abortion': '100000000000011',
    'Fever(F)': '386661006',
    'Water or fluid leakage from eyes,nose or mouth': '64531003',
    'Loose motions': '62315008',
    'Cough': '49727002',
    'blisters on lips': '100000000000102',
    'Stiffening of body': '100000000000112',
    'Fluid leakage from nose': '64531003',
    'Sever cough': '49727002',
    'Pain in stomach and screams': '100000000000113',
    'Loose motions with blood': '62315008',
    'Blood in milk': '100000000000114',
    'Vaginal gnarls': '100000000000115',
    'Gnarls in teats': '100000000000108',
    'Weakness': '100000000000116',
    'Udder become swollen': '100000000000117',
    'Lesion on un_hairy parts of body': '271807003',
    'Swollen under jaw': '100000000000118',
    'Hair fall': '100000000000119',
    'Mites': '100000000000120',
    'Milk in semi solid form': '100000000000104',
    'Mouth frothing': '100000000000100',
    'Pain and Anxiety': '100000000000121',
    'Difficulty in breathing': '267036007',
    'Rapid breathing': '267036007',
    'Shortage of milk': '100000000000101',
    'blisters on feet': '100000000000103',
}

# Try to load SNOMED mapping from external file, fallback to default
SNOMED_MAPPING_FILE = Path("snomed_mapping.json")
if SNOMED_MAPPING_FILE.exists():
    try:
        with open(SNOMED_MAPPING_FILE, 'r', encoding='utf-8') as f:
            snomed_mapping = json.load(f)
            DISEASE_TO_SNOMED = snomed_mapping.get('diseases', _DEFAULT_DISEASE_TO_SNOMED)
            SYMPTOM_TO_SNOMED = snomed_mapping.get('symptoms', _DEFAULT_SYMPTOM_TO_SNOMED)
        print(f"✅ Loaded SNOMED mapping from {SNOMED_MAPPING_FILE}")
    except Exception as e:
        print(f"⚠️  Error loading {SNOMED_MAPPING_FILE}: {e}")
        print("   Using default mappings")
        DISEASE_TO_SNOMED = _DEFAULT_DISEASE_TO_SNOMED
        SYMPTOM_TO_SNOMED = _DEFAULT_SYMPTOM_TO_SNOMED
else:
    print(f"⚠️  {SNOMED_MAPPING_FILE} not found, using default mappings")
    DISEASE_TO_SNOMED = _DEFAULT_DISEASE_TO_SNOMED
    SYMPTOM_TO_SNOMED = _DEFAULT_SYMPTOM_TO_SNOMED


def create_clinical_note(animal_name: str, symptoms: List[str]) -> str:
    """
    Create a natural language clinical note from animal name and symptoms.
    Enhanced to create more realistic veterinary clinical notes.
    
    Args:
        animal_name: Name of the animal (Cow, Buffalo, Goat, Sheep)
        symptoms: List of symptom names that are present
    
    Returns:
        A formatted clinical note string
    """
    if not symptoms:
        return f"{animal_name}. No specific symptoms reported during examination."
    
    # Clean and format symptom names for better readability
    def format_symptom(symptom: str) -> str:
        """Format symptom name to be more natural."""
        # Convert to lowercase and handle special cases
        symptom_lower = symptom.lower()
        
        # Handle common abbreviations and formatting
        replacements = {
            'fever(f)': 'fever',
            'very high fever': 'high fever',
            'continuous loose motions': 'persistent diarrhea',
            'loose motions': 'diarrhea',
            'loose motions with blood': 'bloody diarrhea',
            'blood leakage from nose': 'epistaxis (nosebleed)',
            'difficulty in breathing': 'dyspnea (difficulty breathing)',
            'difficulty in breathing': 'respiratory distress',
            'breathing while mouth open': 'open-mouth breathing',
            'rapid breathing': 'tachypnea (rapid breathing)',
            'mouth frothing': 'oral frothing',
            'drop in milk production': 'decreased milk production',
            'shortage of milk': 'reduced milk yield',
            'milk in semi solid form': 'abnormal milk consistency',
            'blisters in mouth': 'oral vesicles',
            'blisters on lips': 'labial vesicles',
            'blisters on feet': 'foot vesicles',
            'neck swelling': 'cervical swelling',
            'tongue coming out of mouth': 'protruding tongue',
            'teat gnarls': 'teat lesions',
            'gnarls in teats': 'teat lesions',
            'vaginal gnarls': 'vaginal lesions',
            'air/ crapitatifeelings at surface of any body part': 'subcutaneous emphysema',
            'color change of muscles of limb': 'limb muscle discoloration',
            'sudden death': 'acute mortality',
            'water or fluid leakage from eyes,nose or mouth': 'ocular and nasal discharge',
            'fluid leakage from nose': 'nasal discharge',
            'sever cough': 'severe cough',
            'pain in stomach and screams': 'abdominal pain',
            'blood in milk': 'hematuria in milk',
            'udder become swollen': 'udder edema',
            'lesion on un_hairy parts of body': 'skin lesions on glabrous areas',
            'swollen under jaw': 'submandibular swelling',
            'hair fall': 'alopecia',
            'stiffening of body': 'muscle rigidity',
            'pain and anxiety': 'restlessness and discomfort',
        }
        
        for key, value in replacements.items():
            if key in symptom_lower:
                return value
        
        return symptom_lower
    
    # Format symptoms into natural language
    formatted_symptoms = [format_symptom(s) for s in symptoms]
    
    if len(formatted_symptoms) == 1:
        symptoms_text = formatted_symptoms[0]
    elif len(formatted_symptoms) == 2:
        symptoms_text = f"{formatted_symptoms[0]} and {formatted_symptoms[1]}"
    else:
        symptoms_text = ", ".join(formatted_symptoms[:-1]) + f", and {formatted_symptoms[-1]}"
    
    # Create a more natural clinical note
    clinical_note = f"{animal_name}. Clinical presentation includes {symptoms_text}. Physical examination reveals these clinical signs."
    
    return clinical_note


def extract_symptoms_from_row(row: pd.Series, symptom_columns: List[str]) -> List[str]:
    """
    Extract symptoms that are present (value = 1) from a row.
    
    Args:
        row: Pandas Series representing a data row
        symptom_columns: List of column names that represent symptoms
    
    Returns:
        List of symptom names that are present
    """
    symptoms = []
    for col in symptom_columns:
        # Check if symptom is present (value is 1 or True)
        if col in row.index and (row[col] == 1 or row[col] == True or pd.notna(row[col]) and str(row[col]).strip() == '1'):
            symptoms.append(col)
    return symptoms


def get_snomed_codes_for_disease(disease: str) -> List[str]:
    """
    Get SNOMED-CT codes for a given disease.
    
    Args:
        disease: Disease name
    
    Returns:
        List of SNOMED-CT codes
    """
    if pd.isna(disease) or not disease:
        return []
    
    disease_clean = str(disease).strip()
    # Return SNOMED codes only for validated diseases; others will have no codes
    return DISEASE_TO_SNOMED.get(disease_clean, [])


def process_excel_file(file_path: str, output_file: str = None) -> List[Dict[str, Any]]:
    """
    Process an Excel file and convert to JSON format.
    
    Args:
        file_path: Path to the Excel file
        output_file: Optional path to save the JSON output
    
    Returns:
        List of dictionaries in the required format
    """
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"{'='*60}")
    
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Identify columns
    animal_col = 'Animal Name'
    disease_col = 'Disease'
    
    # Get symptom columns (all columns except Animal Name, Disease, and unnamed columns)
    symptom_columns = [col for col in df.columns 
                      if col not in [animal_col, disease_col] 
                      and not str(col).startswith('Unnamed')]
    
    print(f"Found {len(df)} rows")
    print(f"Animal column: {animal_col}")
    print(f"Disease column: {disease_col}")
    print(f"Symptom columns: {len(symptom_columns)}")
    
    # Process each row
    processed_data = []
    skipped_rows = []
    
    for idx, row in df.iterrows():
        try:
            # Extract animal name
            animal_name = str(row[animal_col]).strip() if pd.notna(row[animal_col]) else "Unknown"
            
            # Extract disease
            disease = row[disease_col] if pd.notna(row[disease_col]) else None
            
            # Skip if no disease
            if not disease:
                skipped_rows.append((idx + 2, "No disease specified"))  # +2 for 0-index and header
                continue
            
            # Extract symptoms
            symptoms = extract_symptoms_from_row(row, symptom_columns)
            
            # Skip if no symptoms
            if not symptoms:
                skipped_rows.append((idx + 2, "No symptoms present"))
                continue
            
            # Create clinical note
            clinical_note = create_clinical_note(animal_name, symptoms)
            
            # Get SNOMED-CT codes (only for validated diseases; others get no codes)
            snomed_codes = get_snomed_codes_for_disease(disease)
            if snomed_codes:
                output_codes = ", ".join(snomed_codes)
            else:
                # If no SNOMED mapping, keep the disease label textually
                output_codes = str(disease).strip()
            
            # Create the JSON entry
            entry = {
                "instruction": "Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes.",
                "input": f"Clinical Note: {clinical_note}",
                "output": f"Diagnosed conditions: {output_codes}",
                "snomed_codes": snomed_codes,
                "disease": str(disease).strip(),
                "animal": animal_name,
                "symptoms": symptoms
            }
            
            processed_data.append(entry)
            
        except Exception as e:
            skipped_rows.append((idx + 2, f"Error: {str(e)}"))
            print(f"Warning: Error processing row {idx + 2}: {e}")
            continue
    
    print(f"\n✅ Successfully processed: {len(processed_data)} rows")
    if skipped_rows:
        print(f"⚠️  Skipped: {len(skipped_rows)} rows")
        if len(skipped_rows) <= 10:
            for row_num, reason in skipped_rows:
                print(f"   Row {row_num}: {reason}")
        else:
            print(f"   First 10 skipped rows:")
            for row_num, reason in skipped_rows[:10]:
                print(f"   Row {row_num}: {reason}")
    
    # Save to JSON file if output path provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to: {output_file}")
    
    return processed_data


def main():
    """
    Main function to process all Excel files.
    """
    # Define paths
    dataset_dir = Path("Dataset_UVAS")
    
    # Excel files to process
    excel_files = [
        "Verified DLO data - (Cow Buffalo).xlsx",
        "Verified DLO data (Sheep Goat).xlsx"
    ]
    
    # Output directory
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    all_processed_data = []
    
    # Process each file
    for excel_file in excel_files:
        file_path = dataset_dir / excel_file
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        # Create output filename
        output_filename = excel_file.replace('.xlsx', '_processed.json').replace(' ', '_')
        output_path = output_dir / output_filename
        
        # Process file
        processed_data = process_excel_file(str(file_path), str(output_path))
        all_processed_data.extend(processed_data)
    
    # Create combined output
    if all_processed_data:
        combined_output = output_dir / "all_processed_data.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY")
        print(f"{'='*60}")
        print(f"Total processed entries: {len(all_processed_data)}")
        print(f"Combined output saved to: {combined_output}")
        
        # Show sample entry
        if all_processed_data:
            print(f"\n📋 Sample entry:")
            print(json.dumps(all_processed_data[0], indent=2))
    
    print(f"\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
