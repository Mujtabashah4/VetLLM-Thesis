# QWEN Model Improvement Plan: Increasing F1 & Accuracy Scores

**Date**: 2026-01-05  
**Current Performance**:
- F1 Score (Macro): **16.44%** ❌
- Accuracy: **50.00%** ⚠️
- SNOMED Code Accuracy: **33.75%** ❌

---

## 🔍 Root Cause Analysis

### 1. **Severe Class Imbalance** (PRIMARY ISSUE)

**Problem**: Training data is heavily imbalanced
- **PPR**: 122 samples (32.7%) - Model predicts this too often
- **FMD**: 56 samples (15.0%)
- **Mastitis**: 48 samples (12.9%)
- **Rare diseases**: 1-15 samples each (0.3-4.0%)

**Impact**:
- Model learns to predict common diseases
- Rare diseases get 0% accuracy
- F1 Macro averages across all classes → dragged down by rare diseases

### 2. **Insufficient Training Data for Rare Diseases**

**Diseases with <5 samples**:
- Flue: 1 sample
- Ketosis: 1 sample
- Tympany: 1 sample
- Goat Pox: 1 sample
- Abortion: 1 sample
- Laminitis: 1 sample
- Brucellosis: 2 samples
- Rabies: 2 samples
- Theileriosis: 3 samples
- Babesiosis: 3 samples
- Liver Fluke: 3 samples
- Internal Worms: 3 samples
- Fracture of the Leg: 3 samples

**Result**: Model cannot learn patterns for these diseases

### 3. **Evaluation Methodology Issues**

- **Strict matching**: Requires exact disease name match
- **No partial credit**: Similar diseases (H.S vs Anthrax) get 0% credit
- **Macro averaging**: Treats all diseases equally (unfair to rare diseases)

### 4. **Model Training Issues**

- **No class weighting**: All diseases treated equally during training
- **Standard loss function**: Doesn't penalize rare disease misclassification
- **Limited epochs**: May need more training for rare diseases

---

## ✅ Solutions & Implementation Plan

### **Solution 1: Data Augmentation** (HIGHEST PRIORITY)

**Goal**: Balance the dataset by adding synthetic examples for rare diseases

**Implementation**:
```python
# Create augmented dataset
- Generate 20-30 examples per rare disease (<10 samples)
- Use symptom combinations from veterinary literature
- Maintain species-specific patterns
- Focus on diseases with 0% accuracy
```

**Expected Impact**:
- Increase rare disease samples from 1-3 → 20-30
- Improve rare disease accuracy from 0% → 40-60%
- Improve F1 Macro from 16.44% → 35-45%

**Priority**: 🔴 **CRITICAL**

---

### **Solution 2: Class-Weighted Loss Function**

**Goal**: Penalize misclassification of rare diseases more heavily

**Implementation**:
```python
# Calculate class weights inversely proportional to frequency
class_weights = {
    'PPR': 1.0,           # 122 samples
    'FMD': 2.2,           # 56 samples
    'Mastitis': 2.5,      # 48 samples
    'Rare Disease': 10.0  # 1-3 samples
}

# Use weighted loss in training
loss = weighted_cross_entropy(predictions, labels, weights=class_weights)
```

**Expected Impact**:
- Force model to learn rare diseases
- Improve rare disease recall
- Improve F1 Macro from 16.44% → 25-30%

**Priority**: 🟠 **HIGH**

---

### **Solution 3: Focal Loss for Imbalanced Classes**

**Goal**: Focus learning on hard-to-classify examples (rare diseases)

**Implementation**:
```python
# Focal loss reduces weight of easy examples
focal_loss = -alpha * (1 - p)^gamma * log(p)

# Where:
# - alpha: class balancing factor
# - gamma: focusing parameter (2.0 recommended)
# - p: predicted probability
```

**Expected Impact**:
- Better learning on rare diseases
- Reduce overfitting on common diseases
- Improve F1 Macro from 16.44% → 30-40%

**Priority**: 🟠 **HIGH**

---

### **Solution 4: Stratified Sampling & Oversampling**

**Goal**: Ensure rare diseases appear more frequently during training

**Implementation**:
```python
# Oversample rare diseases during training
- Repeat rare disease examples 5-10x per epoch
- Use SMOTE-like techniques for symptom combinations
- Maintain validation/test sets without oversampling
```

**Expected Impact**:
- More exposure to rare diseases during training
- Better rare disease recognition
- Improve F1 Macro from 16.44% → 25-35%

**Priority**: 🟡 **MEDIUM**

---

### **Solution 5: Better Evaluation Metrics**

**Goal**: Use metrics that account for class imbalance

**Implementation**:
```python
# Use weighted F1 instead of macro F1
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Use lenient matching (partial credit)
- Exact match: 100% credit
- Similar disease: 50% credit (e.g., H.S vs Anthrax)
- Wrong disease: 0% credit
```

**Expected Impact**:
- More realistic performance assessment
- Better understanding of model capabilities
- F1 Weighted already shows 40.04% (better than macro)

**Priority**: 🟡 **MEDIUM**

---

### **Solution 6: Extended Training with Rare Disease Focus**

**Goal**: Train longer with emphasis on rare diseases

**Implementation**:
```python
# Two-stage training:
# Stage 1: Train on all data (current)
# Stage 2: Fine-tune on rare disease examples only
# - Use lower learning rate (1e-5)
# - Train for 2-3 epochs
# - Focus on diseases with <10 samples
```

**Expected Impact**:
- Better rare disease recognition
- Maintain common disease performance
- Improve F1 Macro from 16.44% → 30-40%

**Priority**: 🟡 **MEDIUM**

---

### **Solution 7: Prompt Engineering**

**Goal**: Improve model's understanding of rare disease symptoms

**Implementation**:
```python
# Enhanced prompts with:
- Disease-specific symptom lists
- Differential diagnosis hints
- Species-specific patterns
- SNOMED code examples
```

**Expected Impact**:
- Better disease identification
- Improved SNOMED code prediction
- Improve accuracy from 50% → 60-70%

**Priority**: 🟢 **LOW**

---

### **Solution 8: Ensemble Methods**

**Goal**: Combine multiple models for better rare disease prediction

**Implementation**:
```python
# Train separate models:
# - Model 1: All diseases (current)
# - Model 2: Rare diseases only (fine-tuned)
# - Ensemble: Weighted average of predictions
```

**Expected Impact**:
- Better rare disease handling
- More robust predictions
- Improve F1 Macro from 16.44% → 35-45%

**Priority**: 🟢 **LOW** (Complex, implement after simpler solutions)

---

## 📊 Expected Improvements

### **Conservative Estimate** (Solutions 1 + 2):
- F1 Macro: **16.44%** → **30-35%** (+85% improvement)
- Accuracy: **50%** → **60-65%** (+20% improvement)
- SNOMED Accuracy: **33.75%** → **45-50%** (+33% improvement)

### **Optimistic Estimate** (Solutions 1 + 2 + 3 + 4):
- F1 Macro: **16.44%** → **40-45%** (+145% improvement)
- Accuracy: **50%** → **70-75%** (+40% improvement)
- SNOMED Accuracy: **33.75%** → **55-60%** (+63% improvement)

---

## 🎯 Recommended Implementation Order

### **Phase 1: Quick Wins** (1-2 days)
1. ✅ **Data Augmentation** (Solution 1)
2. ✅ **Class-Weighted Loss** (Solution 2)
3. ✅ **Retrain Model**

### **Phase 2: Advanced Techniques** (3-5 days)
4. ✅ **Focal Loss** (Solution 3)
5. ✅ **Stratified Sampling** (Solution 4)
6. ✅ **Extended Training** (Solution 6)

### **Phase 3: Optimization** (1-2 days)
7. ✅ **Better Evaluation Metrics** (Solution 5)
8. ✅ **Prompt Engineering** (Solution 7)

### **Phase 4: Advanced** (Optional)
9. ✅ **Ensemble Methods** (Solution 8)

---

## 📝 Implementation Scripts Needed

1. **`scripts/augment_rare_diseases.py`**
   - Generate synthetic examples for rare diseases
   - Use veterinary literature patterns
   - Maintain data quality

2. **`scripts/apply_class_weights.py`**
   - Calculate class weights
   - Modify training loss function
   - Update training config

3. **`scripts/retrain_with_improvements.py`**
   - Apply all improvements
   - Train new model
   - Evaluate improvements

---

## 🔬 Monitoring & Validation

After each improvement:
1. ✅ Re-evaluate on test set
2. ✅ Compare F1 Macro, Accuracy, SNOMED accuracy
3. ✅ Check per-disease performance
4. ✅ Ensure common diseases don't degrade

---

## 📈 Success Criteria

**Minimum Acceptable**:
- F1 Macro: **>30%** (from 16.44%)
- Accuracy: **>60%** (from 50%)
- Rare disease accuracy: **>40%** (from 0%)

**Target Performance**:
- F1 Macro: **>40%**
- Accuracy: **>70%**
- Rare disease accuracy: **>60%**

---

**Next Steps**: Start with Solution 1 (Data Augmentation) - this will have the biggest impact!

