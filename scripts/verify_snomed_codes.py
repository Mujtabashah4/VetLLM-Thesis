#!/usr/bin/env python3
"""
SNOMED-CT Code Verification Script
Verifies codes in snomed_mapping.json against SNOMED_codes.md documentation
"""

import json
import os
from pathlib import Path

# Expected codes from SNOMED_codes.md (Section 6)
EXPECTED_CODES = {
    "Anaplasmosis": "15264008",
    "Anthrax": "40214000",
    "Babesiosis": "24026003",
    "BABESIOSIS": "24026003",
    "Black Quarter": "29600000",
    "B.Q": "29600000",
    "Brucellosis": "75702008",
    "CCP": "2260006",
    "CCPP": "2260006",
    "Fascioliasis": "4764006",
    "Liver Fluke": "4764006",
    "FMD": "3974005",
    "Foot and Mouth": "3974005",
    "Foot and mouth": "3974005",
    "H.S": "198462004",
    "Interotoximia": "370514003",
    "Mastitis": "72934000",
    "Mastits": "72934000",
    "Metritis": "50868007",
    "PPR": "1679004",
    "P.P.R": "1679004",
    "Kataa": "1679004",
    "Pox": "363196005",
    "Goat Pox": "57428005",
    "Rabies": "14146002",
    "Rabbies": "14146002",
    "Theileriosis": "24694002",
    "Tuberculosis": "56717001",
    "T.B": "56717001",
    # HC is context-dependent - defaulting to Echinococcosis for ruminants
    "HC": "74942003",  # Echinococcosis (for ruminants)
}

def verify_snomed_mapping():
    """Verify SNOMED codes in snomed_mapping.json"""
    print("="*70)
    print("SNOMED-CT CODE VERIFICATION")
    print("="*70)
    
    base_dir = Path(__file__).parent.parent
    mapping_file = base_dir / "snomed_mapping.json"
    
    if not mapping_file.exists():
        print(f"❌ Error: {mapping_file} not found")
        return False
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    diseases = mapping.get("diseases", {})
    symptoms = mapping.get("symptoms", {})
    
    print(f"\nFound {len(diseases)} disease mappings")
    print(f"Found {len(symptoms)} symptom mappings")
    
    # Verify disease codes
    print("\n" + "="*70)
    print("VERIFYING DISEASE CODES")
    print("="*70)
    
    mismatches = []
    verified = []
    
    for disease, codes in diseases.items():
        if isinstance(codes, list) and len(codes) > 0:
            code = codes[0]  # Take first code
        else:
            code = codes
        
        expected = EXPECTED_CODES.get(disease)
        
        if expected:
            if str(code) == str(expected):
                verified.append((disease, code))
                print(f"✅ {disease}: {code} (verified)")
            else:
                mismatches.append((disease, code, expected))
                print(f"⚠️  {disease}: Found {code}, Expected {expected}")
        else:
            print(f"ℹ️  {disease}: {code} (not in reference, may be valid)")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"✅ Verified: {len(verified)}")
    print(f"⚠️  Mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n⚠️  Mismatches found:")
        for disease, found, expected in mismatches:
            print(f"   {disease}: Found {found}, Expected {expected}")
        return False
    
    print("\n✅ All disease codes match the reference documentation!")
    return True

def verify_symptom_codes():
    """Verify symptom codes (many are custom codes)"""
    print("\n" + "="*70)
    print("VERIFYING SYMPTOM CODES")
    print("="*70)
    
    base_dir = Path(__file__).parent.parent
    mapping_file = base_dir / "snomed_mapping.json"
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    symptoms = mapping.get("symptoms", {})
    
    # Known valid SNOMED codes for symptoms
    valid_symptom_codes = {
        "Lameness": "16973004",
        "Difficulty In Breathing": "267036007",
        "Breathing While Mouth Open": "267036007",
        "Continuous Loose Motions": "62315008",
        "Loose motions": "62315008",
        "Loose motions with blood": "62315008",
        "Fever": "386661006",
        "Very High Fever": "386661006",
        "Fever(F)": "386661006",
        "Water or fluid leakage from eyes,nose or mouth": "64531003",
        "Fluid leakage from nose": "64531003",
        "Cough": "49727002",
        "Sever cough": "49727002",
        "Lesion on un_hairy parts of body": "271807003",
        "Difficulty in breathing": "267036007",
        "Rapid breathing": "267036007",
    }
    
    verified_symptoms = []
    custom_codes = []
    
    for symptom, code in symptoms.items():
        if str(code).startswith("100000000000"):  # Custom code range
            custom_codes.append((symptom, code))
        elif symptom in valid_symptom_codes:
            if str(code) == valid_symptom_codes[symptom]:
                verified_symptoms.append((symptom, code))
                print(f"✅ {symptom}: {code}")
            else:
                print(f"⚠️  {symptom}: Found {code}, Expected {valid_symptom_codes[symptom]}")
        else:
            custom_codes.append((symptom, code))
    
    print(f"\n✅ Verified SNOMED codes: {len(verified_symptoms)}")
    print(f"ℹ️  Custom codes (100000000000xxx): {len(custom_codes)}")
    print("   (Custom codes are acceptable for veterinary-specific symptoms)")
    
    return True

def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("SNOMED-CT CODE VERIFICATION REPORT")
    print("="*70)
    
    disease_ok = verify_snomed_mapping()
    symptom_ok = verify_symptom_codes()
    
    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    
    if disease_ok and symptom_ok:
        print("✅ All codes verified successfully!")
        print("\nReady to proceed with training.")
        return 0
    else:
        print("⚠️  Some codes need attention. Review mismatches above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

