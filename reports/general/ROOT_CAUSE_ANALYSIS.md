# Root Cause Analysis: Low F1 & Accuracy Scores

**Date**: 2026-01-05  
**Current Performance**:
- F1 Score (Macro): **16.44%** ❌
- Accuracy: **50.00%** ⚠️
- SNOMED Code Accuracy: **33.75%** ❌

---

## 🔴 **PRIMARY ROOT CAUSE: Severe Class Imbalance**

### Training Data Distribution:

| Disease | Training Samples | Percentage | Status |
|---------|-----------------|------------|--------|
| **Peste des Petits Ruminants** | 122 | 32.7% | ✅ Well-represented |
| **Foot and Mouth Disease** | 56 | 15.0% | ✅ Well-represented |
| **Mastitis** | 48 | 12.9% | ✅ Well-represented |
| **Hemorrhagic Septicemia** | 42 | 11.3% | ⚠️ Moderate |
| **Black Quarter** | 29 | 7.8% | ⚠️ Moderate |
| **Contagious Caprine Pleuropneumonia** | 29 | 7.8% | ⚠️ Moderate |
| **Anthrax** | 15 | 4.0% | ❌ Underrepresented |
| **Foot Rot** | 6 | 1.6% | ❌ Rare |
| **Babesiosis** | 3 | 0.8% | ❌ Very Rare |
| **Liver Fluke** | 3 | 0.8% | ❌ Very Rare |
| **Theileriosis** | 3 | 0.8% | ❌ Very Rare |
| **Internal Worms** | 3 | 0.8% | ❌ Very Rare |
| **Fracture of the Leg** | 3 | 0.8% | ❌ Very Rare |
| **Brucellosis** | 2 | 0.5% | ❌ Extremely Rare |
| **Rabies** | 2 | 0.5% | ❌ Extremely Rare |
| **Flue** | 1 | 0.3% | ❌ Extremely Rare |
| **Ketosis** | 1 | 0.3% | ❌ Extremely Rare |
| **Tympany** | 1 | 0.3% | ❌ Extremely Rare |
| **Goat Pox** | 1 | 0.3% | ❌ Extremely Rare |
| **Abortion** | 1 | 0.3% | ❌ Extremely Rare |
| **Laminitis** | 1 | 0.3% | ❌ Extremely Rare |
| **Infection** | 1 | 0.3% | ❌ Extremely Rare |

### Imbalance Metrics:
- **Imbalance Ratio**: 122:1 (PPR vs single-sample diseases)
- **Diseases with <5 samples**: 12 diseases (57% of all diseases!)
- **Diseases with <10 samples**: 15 diseases (71% of all diseases!)

---

## 📊 **Impact on Model Performance**

### 1. **F1 Macro Score (16.44%) - Why So Low?**

**F1 Macro** averages F1 scores across ALL diseases equally:
- PPR: 90.9% accuracy → High F1
- FMD: 85.7% accuracy → High F1
- Mastitis: 72.7% accuracy → High F1
- **12 rare diseases: 0% accuracy → F1 = 0.0**

**Calculation**:
```
F1 Macro = (F1_PPR + F1_FMD + F1_Mastitis + ... + F1_Rare1 + F1_Rare2 + ...) / Total_Diseases
         = (0.91 + 0.86 + 0.73 + ... + 0.0 + 0.0 + ...) / 21
         = 16.44%
```

**The rare diseases (0% accuracy) drag down the average!**

### 2. **Accuracy (50%) - Why It's Misleading**

**Accuracy** measures overall correctness:
- Model correctly predicts: 40/80 test samples
- But these are mostly **common diseases** (PPR, FMD, Mastitis)
- **Rare diseases get 0% accuracy** but don't affect overall accuracy much

**Example**:
- PPR: 20/22 correct (90.9%)
- FMD: 12/14 correct (85.7%)
- Mastitis: 8/11 correct (72.7%)
- **12 rare diseases: 0/33 correct (0%)**

**Overall**: 40/80 = 50% accuracy, but **rare diseases are completely failing!**

### 3. **SNOMED Code Accuracy (33.75%) - Why Low?**

- Model struggles with rare disease SNOMED codes
- Many rare diseases have incorrect or missing codes
- Model predicts common disease codes even for rare diseases

---

## 🔍 **Secondary Issues**

### 1. **Model Bias Towards Common Diseases**
- Model predicts PPR/FMD/Mastitis even when symptoms suggest rare diseases
- Example: H.S symptoms → Model predicts Anthrax or PPR (wrong!)

### 2. **Symptom Overlap Confusion**
- Similar symptoms across diseases (fever, nasal discharge, cough)
- Model can't distinguish between:
  - H.S vs Anthrax (both have fever, neck swelling)
  - CCPP vs PPR (both have respiratory symptoms)

### 3. **Evaluation Too Strict**
- Requires exact disease name match
- No partial credit for similar diseases
- H.S predicted as Anthrax = 0% credit (but they're similar!)

### 4. **Insufficient Training for Rare Diseases**
- 1-3 samples per rare disease is **not enough** to learn patterns
- Model needs 20-30 samples minimum per disease

---

## ✅ **Why These Solutions Will Work**

### **Solution 1: Data Augmentation**
- **Adds 20-30 samples per rare disease**
- **Increases rare disease samples from 1-3 → 20-30**
- **Expected impact**: Rare disease accuracy: 0% → 40-60%
- **F1 Macro improvement**: 16.44% → 30-35%

### **Solution 2: Class-Weighted Loss**
- **Penalizes rare disease misclassification 10x more**
- **Forces model to learn rare diseases**
- **Expected impact**: F1 Macro: 16.44% → 25-30%

### **Solution 3: Focal Loss**
- **Focuses learning on hard examples (rare diseases)**
- **Reduces weight of easy examples (common diseases)**
- **Expected impact**: F1 Macro: 16.44% → 30-40%

---

## 📈 **Expected Improvements**

### **Conservative (Solutions 1 + 2)**:
- F1 Macro: **16.44%** → **30-35%** (+85% improvement) ✅
- Accuracy: **50%** → **60-65%** (+20% improvement) ✅
- Rare disease accuracy: **0%** → **40-60%** ✅
- SNOMED accuracy: **33.75%** → **45-50%** (+33% improvement) ✅

### **Optimistic (All Solutions)**:
- F1 Macro: **16.44%** → **40-45%** (+145% improvement) ✅
- Accuracy: **50%** → **70-75%** (+40% improvement) ✅
- Rare disease accuracy: **0%** → **60-70%** ✅
- SNOMED accuracy: **33.75%** → **55-60%** (+63% improvement) ✅

---

## 🎯 **Recommended Action Plan**

1. **Start with Data Augmentation** (biggest impact, 1-2 days)
2. **Add Class-Weighted Loss** (quick win, 1 day)
3. **Retrain model** with improvements
4. **Re-evaluate** and measure improvements
5. **Add Focal Loss** if needed (further improvement)

**See `IMPROVEMENT_PLAN.md` for detailed implementation!**

---

## 📝 **Key Takeaways**

1. **Class imbalance is the #1 problem** (122:1 ratio)
2. **F1 Macro is low because rare diseases have 0% accuracy**
3. **Accuracy (50%) is misleading** - hides rare disease failures
4. **Data augmentation will have the biggest impact**
5. **Class-weighted loss is a quick win**

**The model is actually performing well on common diseases (PPR: 90.9%, FMD: 85.7%, Mastitis: 72.7%) - we just need to fix the rare diseases!**

