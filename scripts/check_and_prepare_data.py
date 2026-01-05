#!/usr/bin/env python3
"""
Data Check and Preparation Script
Checks for data files and helps prepare them if needed
"""

import os
import json
import sys
from pathlib import Path

def check_data_files():
    """Check for existing data files"""
    print("="*70)
    print("DATA FILE CHECK")
    print("="*70)
    
    # Expected locations
    expected_paths = [
        "processed_data/all_processed_data.json",
        "processed_data/Verified_DLO_data_-_(Cow_Buffalo)_processed.json",
        "processed_data/Verified_DLO_data_(Sheep_Goat)_processed.json",
        "data/processed/train_data.json",
        "data/processed/val_data.json",
    ]
    
    found_files = []
    missing_files = []
    
    for path in expected_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
            print(f"✅ Found: {path} ({size:.2f} MB)")
            found_files.append(path)
        else:
            print(f"❌ Missing: {path}")
            missing_files.append(path)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Found: {len(found_files)} files")
    print(f"Missing: {len(missing_files)} files")
    
    if found_files:
        print("\n✅ Data files found! Ready for training.")
        return found_files[0]  # Return first found file
    else:
        print("\n⚠️  No data files found. Need to prepare data.")
        return None

def check_excel_files():
    """Check for Excel source files"""
    print("\n" + "="*70)
    print("CHECKING FOR EXCEL SOURCE FILES")
    print("="*70)
    
    excel_paths = [
        "Dataset_UVAS/Verified DLO data - (Cow Buffalo).xlsx",
        "Dataset_UVAS/Verified DLO data (Sheep Goat).xlsx",
    ]
    
    found_excel = []
    for path in excel_paths:
        if os.path.exists(path):
            print(f"✅ Found Excel: {path}")
            found_excel.append(path)
        else:
            print(f"❌ Missing Excel: {path}")
    
    if found_excel:
        print("\n💡 Excel files found! Run preprocessing:")
        print("   python preprocess_data.py")
        return True
    return False

def main():
    """Main function"""
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)
    
    # Check for data files
    data_file = check_data_files()
    
    # Check for Excel files
    has_excel = check_excel_files()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if data_file:
        print(f"\n✅ Ready to train with: {data_file}")
        print("\nStart training with:")
        print(f"  python scripts/train_vetllm.py --data-path {data_file}")
    elif has_excel:
        print("\n⚠️  Excel files found but not processed.")
        print("Run preprocessing first:")
        print("  python preprocess_data.py")
    else:
        print("\n❌ No data files or Excel sources found.")
        print("Please provide data files in one of these locations:")
        print("  - processed_data/all_processed_data.json")
        print("  - data/processed/train_data.json")
        print("  - Dataset_UVAS/*.xlsx (for preprocessing)")
    
    return 0 if data_file else 1

if __name__ == "__main__":
    sys.exit(main())

