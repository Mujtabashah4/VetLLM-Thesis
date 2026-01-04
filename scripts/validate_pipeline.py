#!/usr/bin/env python3
"""
Pipeline Validation Script
Validates that all required files and dependencies are present for fine-tuning
"""

import os
import sys
import json
import importlib.util
from typing import List, Tuple, Dict

class PipelineValidator:
    """Validates the complete training pipeline"""
    
    REQUIRED_FILES = {
        "scripts/train_vetllm.py": "Main training script",
        "scripts/inference.py": "Inference script",
        "scripts/validate_data.py": "Data validation script",
        "requirements.txt": "Python dependencies",
        "configs/training_config.yaml": "Training configuration",
        "processed_data/all_processed_data.json": "Training data",
    }
    
    OPTIONAL_FILES = {
        "setup.sh": "Setup script",
        "start_training.sh": "Training start script",
        "snomed_mapping.json": "SNOMED code mapping",
        "data/snomed_codes.json": "SNOMED codes reference",
    }
    
    REQUIRED_PACKAGES = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
    ]
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.errors = []
        self.warnings = []
        self.stats = {
            "required_files_present": 0,
            "required_files_missing": 0,
            "optional_files_present": 0,
            "optional_files_missing": 0,
            "packages_installed": 0,
            "packages_missing": 0,
        }
    
    def validate(self) -> Tuple[bool, Dict]:
        """Run all validation checks"""
        print(f"\n{'='*70}")
        print("PIPELINE VALIDATION")
        print(f"{'='*70}")
        
        # Check required files
        self._check_required_files()
        
        # Check optional files
        self._check_optional_files()
        
        # Check Python packages
        self._check_packages()
        
        # Check data files
        self._check_data_files()
        
        # Check configuration
        self._check_configuration()
        
        # Print results
        self._print_results()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.stats
    
    def _check_required_files(self):
        """Check if all required files exist"""
        print(f"\n Checking Required Files:")
        for file_path, description in self.REQUIRED_FILES.items():
            full_path = os.path.join(self.base_dir, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path}")
                self.stats["required_files_present"] += 1
            else:
                print(f"  ✗ {file_path} - MISSING")
                self.errors.append(f"Required file missing: {file_path} ({description})")
                self.stats["required_files_missing"] += 1
    
    def _check_optional_files(self):
        """Check optional files"""
        print(f"\n Checking Optional Files:")
        for file_path, description in self.OPTIONAL_FILES.items():
            full_path = os.path.join(self.base_dir, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path}")
                self.stats["optional_files_present"] += 1
            else:
                print(f"  ⚠ {file_path} - Not found (optional)")
                self.warnings.append(f"Optional file missing: {file_path} ({description})")
                self.stats["optional_files_missing"] += 1
    
    def _check_packages(self):
        """Check if required Python packages are installed"""
        print(f"\n Checking Python Packages:")
        for package in self.REQUIRED_PACKAGES:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    # Try to get version
                    try:
                        mod = importlib.import_module(package)
                        version = getattr(mod, "__version__", "unknown")
                        print(f"  ✓ {package} (version: {version})")
                    except:
                        print(f"  ✓ {package}")
                    self.stats["packages_installed"] += 1
                else:
                    print(f"  ✗ {package} - NOT INSTALLED")
                    self.errors.append(f"Required package missing: {package}")
                    self.stats["packages_missing"] += 1
            except Exception as e:
                print(f"  ✗ {package} - ERROR: {e}")
                self.errors.append(f"Error checking package {package}: {e}")
                self.stats["packages_missing"] += 1
    
    def _check_data_files(self):
        """Check data files"""
        print(f"\n Checking Data Files:")
        data_files = [
            "processed_data/all_processed_data.json",
            "processed_data/Verified_DLO_data_-_(Cow_Buffalo)_processed.json",
            "processed_data/Verified_DLO_data_(Sheep_Goat)_processed.json",
        ]
        
        for data_file in data_files:
            full_path = os.path.join(self.base_dir, data_file)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
                    print(f"  ✓ {data_file} ({len(data)} samples, {file_size:.2f} MB)")
                except Exception as e:
                    print(f"  ✗ {data_file} - Invalid JSON: {e}")
                    self.errors.append(f"Invalid data file: {data_file}")
            else:
                print(f"  ✗ {data_file} - MISSING")
                self.errors.append(f"Data file missing: {data_file}")
    
    def _check_configuration(self):
        """Check configuration files"""
        print(f"\n Checking Configuration:")
        
        # Check training config
        config_path = os.path.join(self.base_dir, "configs/training_config.yaml")
        if os.path.exists(config_path):
            print(f"  ✓ Training config exists")
        else:
            self.warnings.append("Training config file not found")
        
        # Check if scripts are executable
        scripts = ["setup.sh", "start_training.sh"]
        for script in scripts:
            script_path = os.path.join(self.base_dir, script)
            if os.path.exists(script_path):
                is_executable = os.access(script_path, os.X_OK)
                if is_executable:
                    print(f"  ✓ {script} is executable")
                else:
                    self.warnings.append(f"{script} is not executable (run: chmod +x {script})")
    
    def _print_results(self):
        """Print validation results"""
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n Files:")
        print(f"  Required files present: {self.stats['required_files_present']}/{len(self.REQUIRED_FILES)}")
        print(f"  Optional files present: {self.stats['optional_files_present']}/{len(self.OPTIONAL_FILES)}")
        
        print(f"\n Packages:")
        print(f"  Installed: {self.stats['packages_installed']}/{len(self.REQUIRED_PACKAGES)}")
        print(f"  Missing: {self.stats['packages_missing']}")
        
        if self.errors:
            print(f"\n ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        if not self.errors:
            print(f"\n✓ Pipeline validation PASSED! Ready for fine-tuning.")
        else:
            print(f"\n✗ Pipeline validation FAILED. Fix errors before training.")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate training pipeline")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory of the project"
    )
    
    args = parser.parse_args()
    
    validator = PipelineValidator(args.base_dir)
    is_valid, stats = validator.validate()
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())

