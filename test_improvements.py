#!/usr/bin/env python3
"""
Quick test script to verify all improvements are working.
"""
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

def test_post_processing():
    """Test post-processing pipeline."""
    print("=" * 80)
    print("Testing Post-Processing Pipeline")
    print("=" * 80)
    
    try:
        from post_process_codes import SNOMEDCodeProcessor
        
        processor = SNOMEDCodeProcessor()
        
        test_cases = [
            ("Diagnosed conditions: 40214000", "40214000", "Anthrax"),
            ("198462004", "198462004", "H.S"),
            ("The code is 1679004", "1679004", "PPR"),
            ("296000008", "29600000", "B.Q"),  # 9-digit normalized
        ]
        
        passed = 0
        for text, expected_code, expected_disease in test_cases:
            result = processor.process_model_output(text)
            codes = [c['code'] for c in result['validated_codes']]
            diseases = result['diseases']
            
            if expected_code in codes:
                print(f"✅ '{text}' → Extracted: {codes}, Diseases: {diseases}")
                passed += 1
            else:
                print(f"❌ '{text}' → Expected: {expected_code}, Got: {codes}")
        
        print(f"\nPost-Processing: {passed}/{len(test_cases)} tests passed")
        return passed == len(test_cases)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_fuzzy_matching():
    """Test fuzzy matching logic."""
    print("\n" + "=" * 80)
    print("Testing Fuzzy Matching")
    print("=" * 80)
    
    try:
        # Import from validation script
        sys.path.insert(0, str(Path(__file__).parent))
        from comprehensive_validation import VetLLMValidator
        
        validator = VetLLMValidator("dummy", "dummy", use_post_processing=False)
        
        test_cases = [
            ("198462004", "198462004", True),  # Exact match
            ("198462004", "19846200484027004", True),  # Substring match
            ("40214000", "40214000", True),  # Exact match
            ("1679004", "1679004", True),  # Exact match (7-digit)
            ("198462004", "19846200", True),  # Prefix match
            ("40214000", "40214001", False),  # Should not match
        ]
        
        passed = 0
        for expected, predicted, should_match in test_cases:
            result = validator.fuzzy_code_match(expected, predicted)
            if result == should_match:
                print(f"✅ '{expected}' vs '{predicted}' → {result} (expected {should_match})")
                passed += 1
            else:
                print(f"❌ '{expected}' vs '{predicted}' → {result} (expected {should_match})")
        
        print(f"\nFuzzy Matching: {passed}/{len(test_cases)} tests passed")
        return passed == len(test_cases)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VetLLM Improvements Test Suite")
    print("=" * 80)
    
    results = []
    
    # Test 1: Post-processing
    results.append(("Post-Processing", test_post_processing()))
    
    # Test 2: Fuzzy matching
    results.append(("Fuzzy Matching", test_fuzzy_matching()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All improvements are working correctly!")
    else:
        print("\n⚠️  Some improvements need attention.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

