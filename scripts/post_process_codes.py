#!/usr/bin/env python3
"""
Post-processing pipeline for SNOMED-CT code extraction and validation.
Handles code extraction, splitting concatenated codes, and validation.
"""
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

class SNOMEDCodeProcessor:
    """Process and validate SNOMED-CT codes from model outputs."""
    
    # Common SNOMED-CT code prefixes (first 2-3 digits often indicate category)
    KNOWN_CODE_PREFIXES = {
        '40': 'Infectious diseases',
        '16': 'Viral diseases', 
        '19': 'Bacterial diseases',
        '29': 'Musculoskeletal',
        '72': 'Reproductive',
        '39': 'Viral (FMD)',
        '22': 'Respiratory',
        '24': 'Parasitic',
        '75': 'Bacterial (Brucellosis)',
        '14': 'Viral (Rabies)',
        '47': 'Parasitic (Liver Fluke)',
    }
    
    # Expected SNOMED-CT codes from our mapping
    EXPECTED_CODES = {
        '40214000': 'Anthrax',
        '1679004': 'PPR',
        '3974005': 'FMD',
        '198462004': 'H.S',
        '72934000': 'Mastitis',
        '29600000': 'B.Q',
        '2260006': 'CCPP',
        '75702008': 'Brucellosis',
        '24026003': 'Babesiosis',
        '24694002': 'Theileriosis',
        '14146002': 'Rabies',
        '4764006': 'Liver Fluke',
    }
    
    def __init__(self):
        """Initialize processor."""
        self.stats = defaultdict(int)
    
    def extract_codes(self, text: str) -> List[str]:
        """
        Extract SNOMED-CT codes from text.
        Handles various formats and concatenated codes.
        SNOMED-CT codes can be 6-9 digits, but we focus on 7-8 digit codes.
        """
        codes = []
        
        # Pattern 1: Exact 7-9 digit codes (SNOMED-CT format, handle 9-digit as potential 8-digit)
        exact_7_9digit = re.findall(r'\b(\d{7,9})\b', text)
        codes.extend(exact_7_9digit)
        
        # Pattern 2: Codes with punctuation (e.g., "40214000.", "1679004,")
        codes_with_punct = re.findall(r'\b(\d{7,8})[.,;]', text)
        codes.extend(codes_with_punct)
        
        # Pattern 3: Codes in "Diagnosed conditions: 40214000" format
        diagnosed_pattern = re.findall(
            r'(?:diagnosed|diagnosis|code|snomed)[\s:]+(\d{7,8})',
            text, 
            re.IGNORECASE
        )
        codes.extend(diagnosed_pattern)
        
        # Pattern 4: Codes in parentheses or brackets
        codes_in_brackets = re.findall(r'[\(\[{](\d{7,8})[\)\]}]', text)
        codes.extend(codes_in_brackets)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        
        return unique_codes
    
    def split_concatenated_codes(self, text: str) -> List[str]:
        """
        Split concatenated codes intelligently.
        Example: "19846200484027004" -> ["198462004", "84027004"] (if valid)
        """
        # Find all sequences of digits
        digit_sequences = re.findall(r'\d+', text)
        codes = []
        
        for seq in digit_sequences:
            if len(seq) == 8:
                # Perfect 8-digit code
                codes.append(seq)
            elif len(seq) > 8:
                # Potentially concatenated codes
                split_codes = self._split_long_sequence(seq)
                codes.extend(split_codes)
            # Ignore sequences shorter than 8 digits
        
        return codes
    
    def _split_long_sequence(self, sequence: str) -> List[str]:
        """
        Intelligently split a long digit sequence into 8-digit codes.
        Uses heuristics based on known code prefixes.
        """
        codes = []
        remaining = sequence
        
        while len(remaining) >= 8:
            # Try to find a valid code starting from the beginning
            best_split = None
            best_score = 0
            
            # Try different split points (8, 9, 10 digits)
            for length in [8, 9, 10]:
                if len(remaining) >= length:
                    candidate = remaining[:length]
                    score = self._score_code_candidate(candidate)
                    if score > best_score:
                        best_score = score
                        best_split = (candidate[:8], length)  # Take first 8 digits
            
            if best_split and best_score > 0.3:  # Threshold for confidence
                code, length = best_split
                codes.append(code)
                remaining = remaining[length:]
            else:
                # If no good split found, take first 8 digits and continue
                codes.append(remaining[:8])
                remaining = remaining[8:]
        
        return codes
    
    def _score_code_candidate(self, candidate: str) -> float:
        """
        Score a code candidate based on:
        1. If it matches an expected code
        2. If prefix matches known prefixes
        3. If it's exactly 8 digits
        """
        score = 0.0
        
        # Check if it's an exact match
        if candidate[:8] in self.EXPECTED_CODES:
            score += 1.0
        
        # Check prefix
        prefix_2 = candidate[:2]
        prefix_3 = candidate[:3]
        if prefix_2 in self.KNOWN_CODE_PREFIXES:
            score += 0.5
        if prefix_3 in self.KNOWN_CODE_PREFIXES:
            score += 0.3
        
        # Prefer exactly 8 digits
        if len(candidate) == 8:
            score += 0.2
        
        return score
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a SNOMED-CT code.
        Returns (is_valid, disease_name_if_known)
        SNOMED-CT codes can be 6-9 digits, we accept 7-8 digit codes.
        """
        # Normalize: pad 7-digit codes or truncate 9-digit codes
        normalized_code = code
        if len(code) == 7:
            # Try to match known 7-digit codes or pad to 8
            if code in ['1679004']:  # Known 7-digit code
                normalized_code = code
            else:
                normalized_code = code  # Keep as is
        elif len(code) == 9:
            # Try first 8 digits
            normalized_code = code[:8]
        elif len(code) != 8:
            if len(code) < 7 or len(code) > 9:
                return False, None
        
        # Check exact match (try both normalized and original)
        for check_code in [normalized_code, code]:
            if check_code in self.EXPECTED_CODES:
                return True, self.EXPECTED_CODES[check_code]
        
        # Check if code starts with known prefix (partial validation)
        prefix_2 = normalized_code[:2]
        if prefix_2 in self.KNOWN_CODE_PREFIXES:
            return True, None  # Valid format but unknown disease
        
        # Accept 7-8 digit codes as potentially valid
        if 7 <= len(code) <= 8:
            return True, None
        
        return False, None
    
    def process_model_output(self, text: str) -> dict:
        """
        Process model output and extract/validate codes.
        Returns structured result with codes, confidence, etc.
        """
        result = {
            'raw_text': text,
            'extracted_codes': [],
            'validated_codes': [],
            'diseases': [],
            'confidence': 0.0,
            'warnings': []
        }
        
        # Extract codes using multiple methods
        exact_codes = self.extract_codes(text)
        split_codes = self.split_concatenated_codes(text)
        
        # Combine and deduplicate
        all_codes = list(set(exact_codes + split_codes))
        
        # Validate each code
        valid_codes = []
        for code in all_codes:
            is_valid, disease = self.validate_code(code)
            if is_valid:
                valid_codes.append({
                    'code': code,
                    'disease': disease,
                    'confidence': 1.0 if disease else 0.7
                })
                if disease:
                    result['diseases'].append(disease)
        
        result['extracted_codes'] = all_codes
        result['validated_codes'] = valid_codes
        
        # Calculate overall confidence
        if valid_codes:
            result['confidence'] = sum(c['confidence'] for c in valid_codes) / len(valid_codes)
        
        # Add warnings
        if len(all_codes) > len(valid_codes):
            result['warnings'].append(f"Found {len(all_codes)} codes, {len(valid_codes)} valid")
        
        if len(valid_codes) > 3:
            result['warnings'].append(f"Many codes extracted ({len(valid_codes)}), may be concatenated")
        
        return result
    
    def format_output(self, codes: List[str], disease: Optional[str] = None) -> str:
        """
        Format codes for model output.
        Ensures consistent format: "Diagnosed conditions: 40214000"
        """
        if not codes:
            return "Diagnosed conditions: [No valid codes found]"
        
        # Use first valid code (most confident)
        primary_code = codes[0]
        
        if len(codes) == 1:
            return f"Diagnosed conditions: {primary_code}"
        else:
            # Multiple codes - comma separated
            return f"Diagnosed conditions: {', '.join(codes[:3])}"  # Limit to 3 codes


def test_processor():
    """Test the code processor with example outputs."""
    processor = SNOMEDCodeProcessor()
    
    test_cases = [
        "19846200484027004",  # Concatenated
        "Diagnosed conditions: 40214000",  # Correct format
        "198462004.",  # With punctuation
        "The code is 1679004 and also 3974005",  # Multiple codes
        "296000008",  # Close but wrong (9 digits)
        "1984620048402648400000046004000",  # Very long concatenated
    ]
    
    print("=" * 80)
    print("SNOMED-CT Code Processor Test")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test}")
        result = processor.process_model_output(test)
        print(f"  Extracted codes: {result['extracted_codes']}")
        print(f"  Validated codes: {[c['code'] for c in result['validated_codes']]}")
        print(f"  Diseases: {result['diseases']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        if result['warnings']:
            print(f"  Warnings: {', '.join(result['warnings'])}")


if __name__ == "__main__":
    test_processor()

