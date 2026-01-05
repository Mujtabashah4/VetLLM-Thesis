#!/usr/bin/env python3
"""
Comprehensive Data Validation Script for QWEN Fine-tuning Data
Validates the data format, structure, and quality before training
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import re

class QwenDataValidator:
    """Validates QWEN training data format"""
    
    def __init__(self, train_path: str, val_path: str, test_path: str = None):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.errors = []
        self.warnings = []
        self.stats = {
            "train": {},
            "validation": {},
            "test": {},
            "overall": {}
        }
        
    def validate_all(self) -> Tuple[bool, Dict]:
        """Validate all data files"""
        print("\n" + "="*80)
        print("QWEN DATA VALIDATION SUITE")
        print("="*80)
        
        # Validate train data
        print("\n[1/3] Validating Training Data...")
        train_valid, train_stats = self._validate_file(self.train_path, "train")
        self.stats["train"] = train_stats
        
        # Validate validation data
        print("\n[2/3] Validating Validation Data...")
        val_valid, val_stats = self._validate_file(self.val_path, "validation")
        self.stats["validation"] = val_stats
        
        # Validate test data if provided
        if self.test_path and os.path.exists(self.test_path):
            print("\n[3/3] Validating Test Data...")
            test_valid, test_stats = self._validate_file(self.test_path, "test")
            self.stats["test"] = test_stats
        else:
            test_valid = True
            print("\n[3/3] Test data not provided, skipping...")
        
        # Overall validation
        all_valid = train_valid and val_valid and test_valid
        
        # Print summary
        self._print_summary()
        
        return all_valid, self.stats
    
    def _validate_file(self, file_path: str, split_name: str) -> Tuple[bool, Dict]:
        """Validate a single data file"""
        print(f"\n{'='*80}")
        print(f"Validating: {os.path.basename(file_path)} ({split_name})")
        print(f"{'='*80}")
        
        errors = []
        warnings = []
        stats = {
            "file_path": file_path,
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "has_text_field": 0,
            "has_input_field": 0,
            "has_output_field": 0,
            "has_metadata": 0,
            "disease_distribution": {},
            "animal_distribution": {},
            "symptom_count": [],
            "snomed_code_coverage": 0,
            "avg_text_length": 0,
            "avg_input_length": 0,
            "avg_output_length": 0,
        }
        
        # Check file exists
        if not os.path.exists(file_path):
            errors.append(f"File does not exist: {file_path}")
            print(f"❌ ERROR: File not found!")
            return False, stats
        
        # Load JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            print(f"❌ ERROR: Invalid JSON format!")
            return False, stats
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")
            print(f"❌ ERROR: Could not read file!")
            return False, stats
        
        if not isinstance(data, list):
            errors.append("JSON root must be a list/array")
            print(f"❌ ERROR: JSON must be a list!")
            return False, stats
        
        stats["total_samples"] = len(data)
        
        if len(data) == 0:
            errors.append("Dataset is empty")
            print(f"❌ ERROR: Empty dataset!")
            return False, stats
        
        print(f"✓ Found {len(data)} samples")
        
        # Validate each sample
        text_lengths = []
        input_lengths = []
        output_lengths = []
        snomed_samples = 0
        
        for idx, sample in enumerate(data):
            sample_valid = True
            sample_errors = []
            
            # Check required fields for QWEN format
            if "text" not in sample:
                sample_errors.append("Missing 'text' field")
                sample_valid = False
            else:
                stats["has_text_field"] += 1
                text_lengths.append(len(sample["text"]))
                
                # Check text format (should contain QWEN chat template)
                if "<|im_start|>" not in sample["text"]:
                    warnings.append(f"Sample {idx}: 'text' field may not follow QWEN chat template")
            
            if "input" in sample:
                stats["has_input_field"] += 1
                input_lengths.append(len(sample["input"]))
            
            if "output" in sample:
                stats["has_output_field"] += 1
                output_lengths.append(len(sample["output"]))
            
            # Check metadata
            if "metadata" in sample:
                stats["has_metadata"] += 1
                metadata = sample["metadata"]
                
                # Extract disease
                if "disease_normalized" in metadata:
                    disease = metadata["disease_normalized"]
                    stats["disease_distribution"][disease] = stats["disease_distribution"].get(disease, 0) + 1
                
                # Extract animal
                if "animal" in metadata:
                    animal = metadata["animal"]
                    stats["animal_distribution"][animal] = stats["animal_distribution"].get(animal, 0) + 1
                
                # Extract symptoms
                if "symptoms" in metadata and isinstance(metadata["symptoms"], list):
                    stats["symptom_count"].append(len(metadata["symptoms"]))
                
                # Extract SNOMED codes
                if "snomed_codes" in metadata and isinstance(metadata["snomed_codes"], list):
                    if len(metadata["snomed_codes"]) > 0:
                        snomed_samples += 1
            
            if sample_errors:
                errors.extend([f"Sample {idx}: {e}" for e in sample_errors])
                stats["invalid_samples"] += 1
            else:
                stats["valid_samples"] += 1
        
        # Calculate statistics
        if text_lengths:
            stats["avg_text_length"] = sum(text_lengths) / len(text_lengths)
        if input_lengths:
            stats["avg_input_length"] = sum(input_lengths) / len(input_lengths)
        if output_lengths:
            stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        
        if stats["total_samples"] > 0:
            stats["snomed_code_coverage"] = (snomed_samples / stats["total_samples"]) * 100
        
        # Print results
        print(f"\n📊 Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Valid samples: {stats['valid_samples']}")
        print(f"  Invalid samples: {stats['invalid_samples']}")
        print(f"  Has 'text' field: {stats['has_text_field']}/{stats['total_samples']}")
        print(f"  Has 'input' field: {stats['has_input_field']}/{stats['total_samples']}")
        print(f"  Has 'output' field: {stats['has_output_field']}/{stats['total_samples']}")
        print(f"  Has metadata: {stats['has_metadata']}/{stats['total_samples']}")
        print(f"  SNOMED code coverage: {stats['snomed_code_coverage']:.1f}%")
        
        if stats['avg_text_length'] > 0:
            print(f"  Average text length: {stats['avg_text_length']:.1f} chars")
        if stats['avg_input_length'] > 0:
            print(f"  Average input length: {stats['avg_input_length']:.1f} chars")
        if stats['avg_output_length'] > 0:
            print(f"  Average output length: {stats['avg_output_length']:.1f} chars")
        
        # Disease distribution
        if stats["disease_distribution"]:
            print(f"\n📋 Disease Distribution (Top 10):")
            sorted_diseases = sorted(stats["disease_distribution"].items(), key=lambda x: x[1], reverse=True)
            for disease, count in sorted_diseases[:10]:
                print(f"  {disease}: {count}")
        
        # Animal distribution
        if stats["animal_distribution"]:
            print(f"\n🐄 Animal Distribution:")
            for animal, count in sorted(stats["animal_distribution"].items()):
                print(f"  {animal}: {count}")
        
        # Symptom statistics
        if stats["symptom_count"]:
            avg_symptoms = sum(stats["symptom_count"]) / len(stats["symptom_count"])
            print(f"\n📈 Symptom Statistics:")
            print(f"  Average symptoms per sample: {avg_symptoms:.2f}")
            print(f"  Min symptoms: {min(stats['symptom_count'])}")
            print(f"  Max symptoms: {max(stats['symptom_count'])}")
        
        # Print errors
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        
        # Print warnings
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings[:10]:
                print(f"  - {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more warnings")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            print(f"\n✅ {split_name.upper()} DATA VALIDATION PASSED!")
        else:
            print(f"\n❌ {split_name.upper()} DATA VALIDATION FAILED!")
        
        return is_valid, stats
    
    def _print_summary(self):
        """Print overall validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        train_stats = self.stats.get("train", {})
        val_stats = self.stats.get("validation", {})
        test_stats = self.stats.get("test", {})
        
        total_train = train_stats.get("total_samples", 0)
        total_val = val_stats.get("total_samples", 0)
        total_test = test_stats.get("total_samples", 0)
        total_all = total_train + total_val + total_test
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Training samples: {total_train}")
        print(f"  Validation samples: {total_val}")
        print(f"  Test samples: {total_test}")
        print(f"  Total samples: {total_all}")
        
        # Check train/val/test split
        if total_train > 0:
            train_ratio = (total_train / total_all) * 100 if total_all > 0 else 0
            val_ratio = (total_val / total_all) * 100 if total_all > 0 else 0
            test_ratio = (total_test / total_all) * 100 if total_all > 0 else 0
            
            print(f"\n📈 Data Split:")
            print(f"  Train: {train_ratio:.1f}%")
            print(f"  Validation: {val_ratio:.1f}%")
            print(f"  Test: {test_ratio:.1f}%")
            
            # Check if split is reasonable
            if train_ratio < 60:
                self.warnings.append("Training set is less than 60% of total data")
            if val_ratio < 10:
                self.warnings.append("Validation set is less than 10% of total data")
        
        # Overall SNOMED coverage
        train_coverage = train_stats.get("snomed_code_coverage", 0)
        val_coverage = val_stats.get("snomed_code_coverage", 0)
        test_coverage = test_stats.get("snomed_code_coverage", 0)
        
        print(f"\n🏷️  SNOMED Code Coverage:")
        print(f"  Training: {train_coverage:.1f}%")
        print(f"  Validation: {val_coverage:.1f}%")
        print(f"  Test: {test_coverage:.1f}%")
        
        # Overall verdict
        has_errors = len(self.errors) > 0
        
        print("\n" + "="*80)
        if not has_errors:
            print("✅ ALL DATA FILES ARE VALID AND READY FOR FINE-TUNING!")
            print("="*80)
        else:
            print("❌ SOME DATA FILES HAVE ERRORS. Please fix them before training.")
            print("="*80)
        
        if self.warnings:
            print(f"\n⚠️  Note: {len(self.warnings)} warnings found (non-critical)")


def main():
    """Main validation function"""
    # Paths to QWEN data files
    base_dir = Path(__file__).parent
    data_dir = base_dir / "experiments" / "qwen2.5-7b" / "data"
    
    train_path = data_dir / "train.json"
    val_path = data_dir / "validation.json"
    test_path = data_dir / "test.json"
    
    # Check if files exist
    if not train_path.exists():
        print(f"❌ ERROR: Training data not found at {train_path}")
        return 1
    
    if not val_path.exists():
        print(f"❌ ERROR: Validation data not found at {val_path}")
        return 1
    
    # Run validation
    validator = QwenDataValidator(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path) if test_path.exists() else None
    )
    
    is_valid, stats = validator.validate_all()
    
    # Save validation report
    report_path = base_dir / "reports" / "qwen_data_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump({
            "validation_passed": is_valid,
            "stats": stats,
            "errors": validator.errors,
            "warnings": validator.warnings
        }, f, indent=2)
    
    print(f"\n💾 Validation report saved to: {report_path}")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())

