# Evaluation Results Index

**Date**: 2026-01-06  
**Purpose**: Complete index of all evaluation results and metrics for future reference

---

## 📊 QWEN 2.5-7B Evaluation Results

### 1. Comprehensive Evaluation (Test Set: 80 samples)
**File**: `reports/qwen/qwen_comprehensive_evaluation.json`

**Metrics**:
- Accuracy: 50.00%
- Precision (Macro): 15.41%
- Recall (Macro): 19.18%
- F1 Score (Macro): 16.44%
- F1 Score (Micro): 50.00%
- F1 Score (Weighted): 40.04%
- SNOMED Code Accuracy: 33.75% (27/80 correct)

**Top Performing Diseases**:
- PPR: 90.9% (20/22)
- FMD: 85.7% (12/14)
- Mastitis: 72.7% (8/11)

**Report**: `reports/qwen/QWEN_COMPLETE_REPORT.md`

---

### 2. Validation Results (Validation Set: 30 samples)
**File**: `reports/qwen/qwen_validation_results.json`

**Metrics**:
- Accuracy (Strict): 16.67%
- Accuracy (Lenient): 40.00%
- Precision: 21.74%
- Recall: 21.74%
- F1 Score (Strict): 21.74%
- F1 Score (Lenient): 40.00%

**Report**: `reports/qwen/QWEN_VALIDATION_TEST_REPORT.md`

---

### 3. Data Validation
**File**: `reports/qwen/qwen_data_validation.json`

**Dataset Statistics**:
- Training: 373 samples (70%)
- Validation: 80 samples (15%)
- Test: 80 samples (15%)
- Total: 533 samples
- 100% SNOMED code coverage

---

### 4. Inference Test Results
**File**: `reports/qwen/qwen_inference_test.json`

**Test Cases**: 5 sample cases covering:
- PPR (Sheep)
- Anthrax (Cow)
- Hemorrhagic Septicemia (Buffalo)
- Mastitis (Cow)
- CCPP (Goat)

---

## 📊 Alpaca-7B Evaluation Results

### 1. Comprehensive Validation (Validation Set: 30 samples)
**File**: `reports/alpaca/comprehensive_validation_results.json`

**Metrics**:
- Accuracy (Strict): 40.0%
- Accuracy (Lenient): 53.3%
- Precision: 46.15%
- Recall: 46.15%
- F1 Score (Strict): 46.15%
- F1 Score (Lenient): 53.33%

**Per-Disease Performance**:
- FMD: 100% (1/1)
- Mastitis: 100% (2/2)
- Black Quarter: 100% (2/2)
- H.S: 50% (2/4)
- Anthrax: 33% (1/3)

**Report**: `reports/alpaca/ALPACA_COMPLETE_REPORT.md`

---

## 📊 Model Comparison Results

### Comprehensive Comparison
**File**: `reports/comparison/COMPREHENSIVE_MODEL_COMPARISON_REPORT.md`

**Key Metrics Comparison**:
| Metric | Alpaca-7B | QWEN 2.5-7B | Best |
|--------|-----------|-------------|------|
| Accuracy | 40.0% | 50.0% | ✅ QWEN |
| F1 Macro | 46.15% | 16.44% | ✅ Alpaca |
| F1 Weighted | ~50% | 40.04% | ✅ Alpaca |
| SNOMED Accuracy | ~35% | 33.75% | ✅ Alpaca |

**Per-Disease Comparison**:
| Disease | Alpaca-7B | QWEN 2.5-7B | Best |
|---------|-----------|-------------|------|
| PPR | 67% | 90.9% | ✅ QWEN |
| FMD | 100% | 85.7% | ✅ Alpaca |
| Mastitis | 100% | 72.7% | ✅ Alpaca |
| H.S | 50% | 0% | ✅ Alpaca |
| Black Quarter | 100% | 0% | ✅ Alpaca |

---

## 📊 Training Metrics

### Training Metrics
**File**: `reports/general/training_metrics.json`

**QWEN Training**:
- Final Loss: 0.3149 (89% reduction from 2.96)
- Epochs: 5
- Training Time: 7.47 minutes
- Validation Loss: 0.0408

**Alpaca Training**:
- Final Loss: 0.0533 (93.2% reduction from 3.3359)
- Epochs: 3
- Training Time: 10m 26s
- Validation Loss: 0.0533

---

## 📁 File Locations Summary

### QWEN Results
- Comprehensive Evaluation: `reports/qwen/qwen_comprehensive_evaluation.json`
- Validation Results: `reports/qwen/qwen_validation_results.json`
- Data Validation: `reports/qwen/qwen_data_validation.json`
- Inference Test: `reports/qwen/qwen_inference_test.json`

### Alpaca Results
- Validation Results: `reports/alpaca/comprehensive_validation_results.json`

### Comparison
- Full Comparison: `reports/comparison/COMPREHENSIVE_MODEL_COMPARISON_REPORT.md`
- Methodology: `reports/comparison/FAIR_COMPARISON_METHODOLOGY_REPORT.md`
- Thesis Summary: `reports/comparison/THESIS_COMPARISON_SUMMARY.md`

### Training Metrics
- Training Metrics: `reports/general/training_metrics.json`

---

## ✅ All Results Preserved

All evaluation results, metrics, and analysis are preserved in:
- JSON files for programmatic access
- Markdown reports for human reading
- Comparison reports for thesis use

**Nothing was deleted - all results are safe and organized!**

---

**Last Updated**: 2026-01-06

