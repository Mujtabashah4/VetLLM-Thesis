#!/usr/bin/env python3
"""
SNOMED-CT Code Mapping Validation Script
Validates that SNOMED codes in processed data match the mapping files
"""

import json
import os
import sys
from typing import Dict, List, Set, Tuple
from collections import Counter

class SNOMEDValidator:
    """Validates SNOMED-CT code mappings"""
    
    def __init__(self, data_file: str, mapping_file: str = "snomed_mapping.json", 
                 snomed_codes_file: str = "data/snomed_codes.json"):
        self.data_file = data_file
        self.mapping_file = mapping_file
        self.snomed_codes_file = snomed_codes_file
        self.errors = []
        self.warnings = []
        self.stats = {
            "total_samples": 0,
            "samples_with_codes": 0,
            "samples_without_codes": 0,
            "valid_codes": 0,
            "invalid_codes": 0,
            "unmapped_diseases": 0,
            "code_mismatches": 0,
        }
        self.data = None
        self.mapping = None
        self.snomed_codes = None
        
    def load_files(self) -> bool:
        """Load all required files"""
        # Load data file
        if not os.path.exists(self.data_file):
            self.errors.append(f"Data file not found: {self.data_file}")
            return False
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ Loaded data file: {len(self.data)} samples")
        except Exception as e:
            self.errors.append(f"Error loading data file: {e}")
            return False
        
        # Load mapping file
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
                print(f"✓ Loaded mapping file: {self.mapping_file}")
            except Exception as e:
                self.warnings.append(f"Error loading mapping file: {e}")
        else:
            self.warnings.append(f"Mapping file not found: {self.mapping_file}")
        
        # Load SNOMED codes file
        if os.path.exists(self.snomed_codes_file):
            try:
                with open(self.snomed_codes_file, 'r', encoding='utf-8') as f:
                    self.snomed_codes = json.load(f)
                print(f"✓ Loaded SNOMED codes file: {len(self.snomed_codes)} codes")
            except Exception as e:
                self.warnings.append(f"Error loading SNOMED codes file: {e}")
        else:
            self.warnings.append(f"SNOMED codes file not found: {self.snomed_codes_file}")
        
        return True
    
    def validate(self) -> Tuple[bool, Dict]:
        """Run all validation checks"""
        print(f"\n{'='*70}")
        print(f"SNOMED-CT Validation: {os.path.basename(self.data_file)}")
        print(f"{'='*70}")
        
        if not self.load_files():
            return False, self.stats
        
        self.stats["total_samples"] = len(self.data)
        
        # Build code sets
        disease_to_codes = {}
        if self.mapping and "diseases" in self.mapping:
            disease_to_codes = self.mapping["diseases"]
        
        snomed_code_set = set()
        if self.snomed_codes:
            snomed_code_set = {str(code["code_id"]) for code in self.snomed_codes}
        
        # Validate each sample
        for idx, sample in enumerate(self.data):
            disease = sample.get("disease", "").strip()
            snomed_codes = sample.get("snomed_codes", [])
            
            # Count samples with/without codes
            if snomed_codes and len(snomed_codes) > 0:
                self.stats["samples_with_codes"] += 1
                
                # Validate each code
                for code in snomed_codes:
                    code_str = str(code)
                    
                    # Check if code is numeric
                    if not code_str.isdigit():
                        self.errors.append(
                            f"Sample {idx}: Invalid SNOMED code format '{code_str}' (must be numeric)"
                        )
                        self.stats["invalid_codes"] += 1
                    else:
                        # Check if code exists in SNOMED codes file
                        if snomed_code_set and code_str not in snomed_code_set:
                            self.warnings.append(
                                f"Sample {idx}: SNOMED code '{code_str}' not found in snomed_codes.json"
                            )
                        
                        # Check if code matches disease mapping
                        if disease and disease_to_codes:
                            expected_codes = disease_to_codes.get(disease, [])
                            if expected_codes and code_str not in expected_codes:
                                self.warnings.append(
                                    f"Sample {idx}: Disease '{disease}' expects codes {expected_codes}, "
                                    f"but found '{code_str}'"
                                )
                                self.stats["code_mismatches"] += 1
                        
                        self.stats["valid_codes"] += 1
            else:
                self.stats["samples_without_codes"] += 1
                
                # Check if disease should have codes
                if disease and disease_to_codes:
                    if disease in disease_to_codes:
                        expected_codes = disease_to_codes[disease]
                        if expected_codes:
                            self.warnings.append(
                                f"Sample {idx}: Disease '{disease}' should have SNOMED codes "
                                f"({expected_codes}) but snomed_codes is empty"
                            )
                            self.stats["unmapped_diseases"] += 1
        
        # Print results
        self._print_results()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.stats
    
    def _print_results(self):
        """Print validation results"""
        print(f"\n Validation Statistics:")
        print(f"  Total samples: {self.stats['total_samples']}")
        print(f"  Samples with SNOMED codes: {self.stats['samples_with_codes']}")
        print(f"  Samples without SNOMED codes: {self.stats['samples_without_codes']}")
        print(f"  Valid codes found: {self.stats['valid_codes']}")
        print(f"  Invalid codes found: {self.stats['invalid_codes']}")
        print(f"  Code mismatches: {self.stats['code_mismatches']}")
        print(f"  Unmapped diseases: {self.stats['unmapped_diseases']}")
        
        # Code distribution
        if self.data:
            all_codes = []
            for sample in self.data:
                codes = sample.get("snomed_codes", [])
                all_codes.extend([str(c) for c in codes])
            
            if all_codes:
                code_counts = Counter(all_codes)
                print(f"\n Top 10 SNOMED codes:")
                for code, count in code_counts.most_common(10):
                    print(f"  {code}: {count} occurrences")
        
        if self.errors:
            print(f"\n ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:20], 1):
                print(f"  {i}. {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")
        
        if self.warnings:
            print(f"\n WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:20], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")
        
        if not self.errors:
            print(f"\n✓ SNOMED-CT validation passed!")
        else:
            print(f"\n✗ SNOMED-CT validation failed with errors.")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate SNOMED-CT code mappings")
    parser.add_argument(
        "--data-file",
        default="processed_data/all_processed_data.json",
        help="Path to processed data file"
    )
    parser.add_argument(
        "--mapping-file",
        default="snomed_mapping.json",
        help="Path to SNOMED mapping file"
    )
    parser.add_argument(
        "--snomed-codes-file",
        default="data/snomed_codes.json",
        help="Path to SNOMED codes file"
    )
    
    args = parser.parse_args()
    
    validator = SNOMEDValidator(
        args.data_file,
        args.mapping_file,
        args.snomed_codes_file
    )
    
    is_valid, stats = validator.validate()
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())

