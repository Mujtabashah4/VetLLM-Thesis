# How to Improve Disease Prediction Accuracy

**Date:** January 5, 2026  
**Focus:** Rare Diseases & Small Animals

---

## 📊 Current Issues

### 1. Rare Diseases (0% accuracy):
- **CCPP**: 0/2 (but has 81 training samples!)
- **Brucellosis**: 0/1 (7 training samples)
- **Babesiosis**: 0/1 (9 training samples)
- **Theileriosis**: 0/1 (4 training samples)
- **Rabies**: 0/1 (4 training samples)
- **Liver Fluke**: 0/1 (4 training samples)
- **Internal Worms**: 0/1 (4 training samples)
- **Foot Rot**: 0/1 (24 training samples)

### 2. Small Animals (Low accuracy):
- **Goat**: 25% (2/8 correct)
- **Sheep**: 20% (1/5 correct)

---

## 🔍 Root Cause Analysis

### Issue 1: CCPP Has Data But Model Fails
- **Training samples**: 81 (plenty!)
- **Problem**: Model predicts PPR (1679004) instead of CCPP (2260006)
- **Reason**: Symptoms overlap (cough, fever, nasal discharge)
- **Solution**: Need more distinct examples highlighting CCPP-specific symptoms

### Issue 2: Rare Diseases Have Few Samples
- Most rare diseases have < 10 training samples
- Model hasn't learned their patterns
- **Solution**: Add more training examples

### Issue 3: Small Animals Underperform
- Goat: 421 samples (good)
- Sheep: 435 samples (good)
- **Problem**: Model trained but not predicting correctly
- **Reason**: May need more diverse examples or better symptom combinations

---

## ✅ Solutions Implemented

### 1. Data Augmentation Script Created
**File**: `scripts/augment_training_data.py`

**What it does**:
- Generates 25 examples for diseases with 0 samples
- Adds examples for diseases with < 10 samples
- Creates diverse symptom combinations
- Focuses on small animals (Goat, Sheep)

**Generated**:
- ✅ Rabies: 25 examples
- ✅ Brucellosis: +8 examples
- ✅ Babesiosis: +6 examples
- ✅ Theileriosis: +11 examples
- ✅ Liver Fluke: +11 examples
- ✅ Internal Worms: +16 examples
- ✅ Mites: +13 examples
- ✅ Ketosis: +13 examples
- ✅ Tympany: +13 examples
- ✅ Fracture: +16 examples

**Total**: +132 new samples → 1,734 total

---

## 🎯 Next Steps to Improve

### Step 1: Review Augmented Data
```bash
# Check the augmented data
python3 -c "
import json
with open('processed_data/all_processed_data_augmented.json', 'r') as f:
    data = json.load(f)
print(f'Total samples: {len(data)}')

# Count by disease
diseases = {}
for item in data:
    d = item.get('disease', '').lower()
    diseases[d] = diseases.get(d, 0) + 1

for d, c in sorted(diseases.items()):
    if c < 15:
        print(f'{d}: {c}')
"
```

### Step 2: Update Training Script
Modify `scripts/train_vetllm_improved.py`:
```python
# Change this line:
data_path: str = "processed_data/all_processed_data.json"

# To:
data_path: str = "processed_data/all_processed_data_augmented.json"
```

### Step 3: Retrain Model
```bash
# Remove old model (optional)
rm -rf models/vetllm-finetuned-continued

# Start training with augmented data
source venv/bin/activate
python3 scripts/train_vetllm_improved.py
```

### Step 4: Validate Improvements
```bash
# Test the new model
python3 comprehensive_validation.py --model-path models/vetllm-finetuned-continued
```

---

## 🔧 Additional Improvements Needed

### For CCPP (Has data but fails):
1. **Add more distinct examples**:
   - Emphasize CCPP-specific symptoms (severe respiratory distress, rapid breathing)
   - Contrast with PPR (CCPP is more respiratory-focused)
   - Add examples: "Goat with severe cough and rapid breathing" → CCPP

2. **Improve prompt**:
   - Make model focus on respiratory symptoms
   - Add examples where CCPP is clearly different from PPR

### For Small Animals:
1. **Add more diverse symptom combinations**:
   - Different symptom orders
   - Various severity levels
   - Multiple animals per disease

2. **Balance animal distribution**:
   - Ensure Goat/Sheep examples cover all diseases
   - Add edge cases specific to small animals

### For Rare Diseases:
1. **Collect real clinical data** (if possible):
   - Real veterinary cases
   - Diverse symptom presentations
   - Multiple animals

2. **Synthetic data augmentation**:
   - Use the augmentation script
   - Generate 20-30 examples per rare disease
   - Ensure symptom diversity

---

## 📋 Manual Data Collection Template

If you have access to real veterinary data, use this format:

```json
{
  "instruction": "Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes.",
  "input": "Clinical Note: [Animal]. Clinical presentation includes [symptom1], [symptom2], [symptom3]. Physical examination reveals these clinical signs.",
  "output": "Diagnosed conditions: [SNOMED_CODE]",
  "disease": "[Disease Name]",
  "animal": "[Animal Type]",
  "symptoms": ["symptom1", "symptom2", "symptom3"]
}
```

**Priority diseases to collect**:
1. CCPP (distinct from PPR)
2. Brucellosis
3. Babesiosis
4. Theileriosis
5. Rabies
6. Liver Fluke

---

## 💡 Expected Improvements

After retraining with augmented data:

| Metric | Current | Expected |
|--------|---------|----------|
| **Rare Disease Accuracy** | 0-33% | 50-70% |
| **Goat Accuracy** | 25% | 40-50% |
| **Sheep Accuracy** | 20% | 40-50% |
| **Overall Accuracy** | 53.3% | 60-65% |

---

## 🚀 Quick Start

```bash
# 1. Generate augmented data (already done)
python3 scripts/augment_training_data.py

# 2. Update training script to use augmented data
# Edit scripts/train_vetllm_improved.py line 88:
# data_path: str = "processed_data/all_processed_data_augmented.json"

# 3. Retrain
python3 scripts/train_vetllm_improved.py

# 4. Validate
python3 comprehensive_validation.py
```

---

**Status**: ✅ Augmentation script ready. Ready to retrain with improved data!

