#!/usr/bin/env python3
"""
Comprehensive Data Validation Script for VetLLM Training Data
Validates JSON structure, format consistency, and data integrity
"""

import json
import os
import sys
from typing import Dict, List, Any, Tuple
from collections import Counter
import re

class DataValidator:
    """Validates veterinary training data for fine-tuning"""
    
    REQUIRED_FIELDS = ["instruction", "input", "output"]
    OPTIONAL_FIELDS = ["snomed_codes", "disease", "animal", "symptoms"]
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.errors = []
        self.warnings = []
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "empty_snomed_codes": 0,
            "inconsistent_outputs": 0,
            "duplicate_samples": 0,
        }
        self.data = None
        
    def validate(self) -> Tuple[bool, Dict]:
        """Run all validation checks"""
        print(f"\n{'='*70}")
        print(f"Validating: {os.path.basename(self.file_path)}")
        print(f"{'='*70}")
        
        # Step 1: Check file exists and is readable
        if not self._check_file_exists():
            return False, self.stats
            
        # Step 2: Validate JSON structure
        if not self._validate_json_structure():
            return False, self.stats
            
        # Step 3: Validate each sample
        self._validate_samples()
        
        # Step 4: Check for duplicates
        self._check_duplicates()
        
        # Step 5: Check output format consistency
        self._check_output_consistency()
        
        # Step 6: Check SNOMED code format
        self._check_snomed_codes()
        
        # Step 7: Validate data types
        self._validate_data_types()
        
        # Step 8: Check for empty/null values
        self._check_empty_values()
        
        # Print results
        self._print_results()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.stats
    
    def _check_file_exists(self) -> bool:
        """Check if file exists and is readable"""
        if not os.path.exists(self.file_path):
            self.errors.append(f"File does not exist: {self.file_path}")
            return False
        
        if not os.path.isfile(self.file_path):
            self.errors.append(f"Path is not a file: {self.file_path}")
            return False
        
        return True
    
    def _validate_json_structure(self) -> bool:
        """Validate JSON structure and parse"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            if not isinstance(self.data, list):
                self.errors.append("JSON root must be a list/array")
                return False
            
            self.stats["total_samples"] = len(self.data)
            
            if len(self.data) == 0:
                self.errors.append("Dataset is empty - no samples found")
                return False
            
            print(f" JSON structure valid - {len(self.data)} samples found")
            return True
            
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON format: {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading file: {str(e)}")
            return False
    
    def _validate_samples(self):
        """Validate each sample in the dataset"""
        for idx, sample in enumerate(self.data):
            sample_errors = []
            
            # Check required fields
            for field in self.REQUIRED_FIELDS:
                if field not in sample:
                    sample_errors.append(f"Missing required field: '{field}'")
                elif sample[field] is None:
                    sample_errors.append(f"Field '{field}' is None")
                elif not isinstance(sample[field], str):
                    sample_errors.append(f"Field '{field}' must be a string, got {type(sample[field]).__name__}")
                elif field == "instruction" and len(sample[field].strip()) == 0:
                    sample_errors.append(f"Field 'instruction' cannot be empty")
                elif field == "output" and len(sample[field].strip()) == 0:
                    sample_errors.append(f"Field 'output' cannot be empty")
            
            # Check instruction format
            if "instruction" in sample and isinstance(sample["instruction"], str):
                expected_instruction = "Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes."
                if sample["instruction"] != expected_instruction:
                    self.warnings.append(
                        f"Sample {idx}: Instruction doesn't match expected format"
                    )
            
            # Check input format
            if "input" in sample and isinstance(sample["input"], str):
                if not sample["input"].startswith("Clinical Note:"):
                    self.warnings.append(
                        f"Sample {idx}: Input doesn't start with 'Clinical Note:'"
                    )
            
            if sample_errors:
                self.errors.append(f"Sample {idx}: {'; '.join(sample_errors)}")
                self.stats["invalid_samples"] += 1
            else:
                self.stats["valid_samples"] += 1
    
    def _check_duplicates(self):
        """Check for duplicate samples"""
        seen = set()
        duplicates = []
        
        for idx, sample in enumerate(self.data):
            # Create a hashable representation
            sample_key = (
                sample.get("instruction", ""),
                sample.get("input", ""),
                sample.get("output", "")
            )
            
            if sample_key in seen:
                duplicates.append(idx)
            else:
                seen.add(sample_key)
        
        if duplicates:
            self.warnings.append(f"Found {len(duplicates)} duplicate samples (indices: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''})")
            self.stats["duplicate_samples"] = len(duplicates)
    
    def _check_output_consistency(self):
        """Check output format consistency"""
        output_patterns = {
            "diagnosed_conditions": re.compile(r"Diagnosed conditions:\s*([\d\s,]+)", re.IGNORECASE),
            "diagnosed_conditions_text": re.compile(r"Diagnosed conditions:\s*([A-Za-z\s,]+)", re.IGNORECASE),
            "plain_text": re.compile(r"^[A-Za-z\s]+$"),
        }
        
        inconsistent = []
        
        for idx, sample in enumerate(self.data):
            output = sample.get("output", "")
            snomed_codes = sample.get("snomed_codes", [])
            
            # Check if output matches snomed_codes
            if isinstance(snomed_codes, list) and len(snomed_codes) > 0:
                # Extract codes from output
                match = output_patterns["diagnosed_conditions"].search(output)
                if match:
                    output_codes = [c.strip() for c in match.group(1).split(",")]
                    output_codes = [c for c in output_codes if c]
                    
                    # Compare with snomed_codes field
                    snomed_str = [str(c) for c in snomed_codes]
                    if set(output_codes) != set(snomed_str):
                        inconsistent.append(idx)
                        self.stats["inconsistent_outputs"] += 1
                else:
                    # Output doesn't contain codes but snomed_codes is not empty
                    inconsistent.append(idx)
                    self.stats["inconsistent_outputs"] += 1
            elif isinstance(snomed_codes, list) and len(snomed_codes) == 0:
                # No SNOMED codes - output should be text only
                if not output_patterns["diagnosed_conditions_text"].search(output) and not output_patterns["plain_text"].match(output.strip()):
                    # Check if it says "Diagnosed conditions:" with no codes
                    if "Diagnosed conditions:" in output:
                        # This is acceptable - disease name without code
                        pass
                    else:
                        self.warnings.append(
                            f"Sample {idx}: Empty snomed_codes but output format may be inconsistent"
                        )
                self.stats["empty_snomed_codes"] += 1
        
        if inconsistent:
            self.warnings.append(
                f"Found {len(inconsistent)} samples with inconsistent output/snomed_codes (indices: {inconsistent[:10]}{'...' if len(inconsistent) > 10 else ''})"
            )
    
    def _check_snomed_codes(self):
        """Validate SNOMED code format"""
        snomed_pattern = re.compile(r"^\d+$")
        
        for idx, sample in enumerate(self.data):
            snomed_codes = sample.get("snomed_codes", [])
            
            if snomed_codes is not None:
                if not isinstance(snomed_codes, list):
                    self.errors.append(f"Sample {idx}: 'snomed_codes' must be a list")
                else:
                    for code in snomed_codes:
                        if not isinstance(code, (str, int)):
                            self.errors.append(
                                f"Sample {idx}: SNOMED code must be string or int, got {type(code).__name__}"
                            )
                        else:
                            code_str = str(code)
                            if not snomed_pattern.match(code_str):
                                self.errors.append(
                                    f"Sample {idx}: Invalid SNOMED code format: '{code_str}' (must be numeric)"
                                )
    
    def _validate_data_types(self):
        """Validate data types for optional fields"""
        for idx, sample in enumerate(self.data):
            # Check disease field
            if "disease" in sample and sample["disease"] is not None:
                if not isinstance(sample["disease"], str):
                    self.errors.append(
                        f"Sample {idx}: 'disease' must be a string, got {type(sample['disease']).__name__}"
                    )
            
            # Check animal field
            if "animal" in sample and sample["animal"] is not None:
                if not isinstance(sample["animal"], str):
                    self.errors.append(
                        f"Sample {idx}: 'animal' must be a string, got {type(sample['animal']).__name__}"
                    )
            
            # Check symptoms field
            if "symptoms" in sample and sample["symptoms"] is not None:
                if not isinstance(sample["symptoms"], list):
                    self.errors.append(
                        f"Sample {idx}: 'symptoms' must be a list, got {type(sample['symptoms']).__name__}"
                    )
                else:
                    for symptom in sample["symptoms"]:
                        if not isinstance(symptom, str):
                            self.errors.append(
                                f"Sample {idx}: Each symptom must be a string, got {type(symptom).__name__}"
                            )
    
    def _check_empty_values(self):
        """Check for problematic empty values"""
        for idx, sample in enumerate(self.data):
            # Check for empty strings in required fields
            if "input" in sample:
                if sample["input"] is not None and len(sample["input"].strip()) == 0:
                    # Empty input is acceptable (some samples may not have input)
                    pass
            
            # Check for None values in optional fields
            for field in self.OPTIONAL_FIELDS:
                if field in sample and sample[field] is None:
                    # None is acceptable for optional fields, but empty list is better
                    if field == "snomed_codes" or field == "symptoms":
                        self.warnings.append(
                            f"Sample {idx}: '{field}' is None, should be empty list []"
                        )
    
    def _print_results(self):
        """Print validation results"""
        print(f"\n Validation Statistics:")
        print(f"  Total samples: {self.stats['total_samples']}")
        print(f"  Valid samples: {self.stats['valid_samples']}")
        print(f"  Invalid samples: {self.stats['invalid_samples']}")
        print(f"  Empty SNOMED codes: {self.stats['empty_snomed_codes']}")
        print(f"  Inconsistent outputs: {self.stats['inconsistent_outputs']}")
        print(f"  Duplicate samples: {self.stats['duplicate_samples']}")
        
        if self.errors:
            print(f"\n ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:20], 1):  # Show first 20 errors
                print(f"  {i}. {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")
        
        if self.warnings:
            print(f"\n️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:20], 1):  # Show first 20 warnings
                print(f"  {i}. {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")
        
        if not self.errors and not self.warnings:
            print(f"\n All checks passed! Data is ready for fine-tuning.")
        elif not self.errors:
            print(f"\n️  Data has warnings but no critical errors. Review warnings before training.")
        else:
            print(f"\n Data has critical errors. Fix them before training.")

def main():
    """Main validation function"""
    # Data files to validate
    data_files = [
        "processed_data/all_processed_data.json",
        "processed_data/Verified_DLO_data_-_(Cow_Buffalo)_processed.json",
        "processed_data/Verified_DLO_data_(Sheep_Goat)_processed.json",
    ]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    all_valid = True
    all_stats = {}
    
    for data_file in data_files:
        file_path = os.path.join(base_dir, data_file)
        
        if not os.path.exists(file_path):
            print(f"\n File not found: {file_path}")
            all_valid = False
            continue
        
        validator = DataValidator(file_path)
        is_valid, stats = validator.validate()
        all_valid = all_valid and is_valid
        all_stats[os.path.basename(file_path)] = stats
    
    # Print summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    total_samples = sum(stats["total_samples"] for stats in all_stats.values())
    total_valid = sum(stats["valid_samples"] for stats in all_stats.values())
    
    print(f"Total files validated: {len(all_stats)}")
    print(f"Total samples across all files: {total_samples}")
    print(f"Total valid samples: {total_valid}")
    
    if all_valid:
        print(f"\n ALL FILES ARE VALID AND READY FOR FINE-TUNING!")
        return 0
    else:
        print(f"\n SOME FILES HAVE ERRORS. Please fix them before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

