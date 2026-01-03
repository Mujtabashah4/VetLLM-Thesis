# MS Thesis Defense Proposal

## VetLLM: Deep Learning-Based Livestock Disease Outbreak Prediction System for Early Clinical Decision Support

---

**Candidate:** Syed Muhammad Mujtaba  
**Roll Number:** 24280069  
**Degree:** Master of Science in Artificial Intelligence  
**Supervisor:** Dr. Malik Jahan Khan  
**Institution:** Lahore University of Management Sciences (LUMS), Lahore  
**Academic Year:** 2024-2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction and Background](#2-introduction-and-background)
3. [Problem Statement](#3-problem-statement)
4. [Research Objectives](#4-research-objectives)
5. [Literature Review](#5-literature-review)
6. [Research Gap Analysis](#6-research-gap-analysis)
7. [Proposed Methodology](#7-proposed-methodology)
8. [Dataset Description](#8-dataset-description)
9. [Expected Results](#9-expected-results)
10. [Research Contributions](#10-research-contributions)
11. [Timeline and Milestones](#11-timeline-and-milestones)
12. [Limitations and Mitigation](#12-limitations-and-mitigation)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Executive Summary

This thesis proposes **VetLLM**, a deep learning-based livestock disease outbreak prediction system designed for early warning and clinical decision support in Pakistani veterinary contexts. Pakistan's livestock sector, comprising approximately **212 million animals** and contributing **2.3% to national GDP**, faces annual losses exceeding **\$500 million** due to preventable diseases. The 2021-2022 Lumpy Skin Disease outbreak, which infected over 36,000 cattle with a 4-month detection delay, exemplifies the critical need for proactive disease surveillance.

Our approach addresses four fundamental limitations of existing methods:
1. Data scarcity in veterinary AI
2. Multi-label classification complexity
3. Absence of temporal symptom modeling
4. Lack of interpretable clinical outputs

VetLLM combines LSTM-based temporal modeling, weighted multi-label loss functions, and multi-task species-adaptive learning to achieve early disease prediction 3-7 days before clinical manifestation.

**Key Expected Outcomes:**
- **Macro F1 Score:** 0.89 (+9.9% over baseline)
- **Rare Disease Detection:** +40% improvement
- **Early Warning Lead Time:** 3-7 days
- **Veterinarian Agreement:** >80%

---

## 2. Introduction and Background

### 2.1 Pakistan's Livestock Crisis

Pakistan possesses one of the world's largest livestock populations, with approximately **212 million animals** generating:
- **2.3% of national GDP**
- **15-20% of agricultural GDP**
- Livelihoods for **30+ million rural families**

However, this critical sector faces an escalating disease management crisis that threatens agricultural sustainability, public health, and economic stability.

### 2.2 The Disease Surveillance Failure

**Case Study: Lumpy Skin Disease Outbreak (2021-2022)**

| Timeline | Event |
|----------|-------|
| November 2021 | First case in Jamshoro district, Sindh |
| March 2022 | Official detection and notification |
| **Detection Delay** | **4 months** |
| Animals Infected | 36,000+ cattle |
| Mortality | 168 confirmed deaths (0.8%) |
| Economic Impact | \$100M+ in 6 months |

This outbreak exemplifies a systemic failure: Pakistan's disease surveillance is fundamentally **reactive rather than proactive**.

### 2.3 Current Disease Landscape

| Disease | Status | Impact |
|---------|--------|--------|
| **Foot and Mouth Disease (FMD)** | Highly endemic | \$500M annual losses |
| **Lumpy Skin Disease (LSD)** | Emerging | Milk production loss |
| **Brucellosis** | 3-8.5% prevalence | Zoonotic risk |
| **Hemorrhagic Septicemia (HS)** | High mortality | Significant buffalo deaths |

### 2.4 Why AI/ML is the Solution

Traditional surveillance fails because:
1. **Cognitive Limitations:** Processing 15+ symptom variables exceeds human capacity
2. **Temporal Constraints:** Manual systems cannot achieve early detection
3. **Data Integration:** No mechanism aggregates observations
4. **Predictive Incapacity:** Traditional approaches are reactive

---

## 3. Problem Statement

### 3.1 Core Problem

> **Veterinary medicine currently lacks automated, clinically-integrated systems to predict livestock diseases from clinical symptoms with accuracy and interpretability sufficient for early warning and clinical decision support.**

### 3.2 Technical Challenges

1. **Data Scarcity Paradox:** Manual annotation costs \$50-200 per case; traditional learning requires 50,000+ examples
2. **Multi-Label Complexity:** Animals present with 2-4 concurrent diagnoses
3. **Extreme Class Imbalance:** Rare diseases <0.1% but clinically critical
4. **Temporal Dynamics Ignored:** No existing systems model symptom progression

### 3.3 Formal Problem Definition

**Given:** Dataset D = {(xi, yi)} where xi represents symptom vector, yi represents multi-hot disease labels

**Objective:** Learn function f: symptoms → disease probabilities that:
- Handles multi-label concurrent diseases
- Incorporates temporal symptom progression
- Provides interpretable predictions
- Generalizes across species

---

## 4. Research Objectives

### 4.1 Primary Objectives

| # | Objective | Target |
|---|-----------|--------|
| 1 | Technical Performance | >85% F1 common, >70% F1 rare diseases |
| 2 | Early Warning | 3-7 day prediction lead time |
| 3 | Data Efficiency | <1,000 training examples |
| 4 | Cross-Species Generalization | <10% F1 degradation |
| 5 | Clinical Interpretability | >80% veterinarian agreement |

---

## 5. Literature Review

### 5.1 Traditional Veterinary Surveillance

- **Passive Surveillance:** 30-50% outbreak detection rate, days-to-weeks lag
- **Limitations:** No pattern analysis, judgment-dependent, no prediction

### 5.2 Machine Learning in Medical Diagnosis

| Study | Method | Performance | Limitation |
|-------|--------|-------------|------------|
| DeepTag (2018) | Neural networks | 65-70% F1 | Required 100K+ examples |
| VetTag (2019) | Enhanced NN | 74.7% F1 | Data-intensive |
| Saqib et al. (2024) | MobileNetV2 | 95% accuracy | Post-clinical only |
| NIH Study (2024) | CNN ensemble | 96% accuracy | Single disease |

**Critical Gap:** All existing approaches are image-based (post-clinical), single-disease, or data-intensive.

### 5.3 Key Technical Foundations

**LSTM (Long Short-Term Memory):**
- Forget gate: ft = σ(Wf[ht-1, xt] + bf)
- Input gate: it = σ(Wi[ht-1, xt] + bi)
- Cell state: Ct = ft ⊙ Ct-1 + it ⊙ C̃t
- Output: ht = ot ⊙ tanh(Ct)

**Focal Loss:**
L_focal = -Σ(1-pt)^γ log(pt)

**SHAP Interpretability:**
φj = Σ [|S|!(|F|-|S|-1)!/|F|!] × [f(S∪{j}) - f(S)]

---

## 6. Research Gap Analysis

| Gap | Previous Work | Our Contribution |
|-----|---------------|------------------|
| Dataset | Synthetic/Western | Authenticated Pakistani UVAS data |
| Task | Single-disease | Multi-label (1-3 diseases) |
| Temporal | Static features | LSTM sequence modeling |
| Species | Single/separate | Multi-task learning |
| Rare Disease | Optimized for common | Weighted loss for critical rare |
| Interpretability | Black-box | SHAP clinical explanations |

---

## 7. Proposed Methodology

### 7.1 System Architecture

```
Input: Symptom Sequence [xt-2, xt-1, xt]
           ↓
┌─────────────────────────────┐
│   Shared Symptom Encoder    │
│   Dense(15→128→64) + ReLU   │
└─────────────────────────────┘
           ↓
┌─────────────────────────────┐
│   LSTM Temporal Module      │
│   64 units, dropout 0.2     │
└─────────────────────────────┘
           ↓
┌────────┬────────┬────────┬────────┐
│ Cattle │Buffalo │ Sheep  │ Goat   │
│  Head  │  Head  │  Head  │  Head  │
└────────┴────────┴────────┴────────┘
           ↓
Output: Disease Probabilities (Sigmoid)
```

### 7.2 Novel Weighted Multi-Label Loss

**L_VetLLM = -(1/N) Σi Σj wj [yij(1-ŷij)^γ log(ŷij) + (1-yij)ŷij^γ log(1-ŷij)]**

- **wj = 1/pj:** Inverse prevalence weighting (rare diseases get 50x more weight)
- **γ = 2:** Focal parameter (focuses on hard examples)

### 7.3 Multi-Task Species Learning

**L_total = Σs λs Ls** where λs = ns/N

Shared encoder learns general patterns; species-specific heads adapt predictions.

### 7.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (weight decay 0.01) |
| Learning Rate | 1×10⁻³ with cosine annealing |
| Batch Size | 32 |
| Epochs | 100 (early stopping patience=10) |
| Cross-Validation | 5-fold stratified |

---

## 8. Dataset Description

### 8.1 Overview

| Attribute | Value |
|-----------|-------|
| Source | UVAS, Lahore |
| Total Animals | 1,050 |
| Species | 4 (Cattle, Buffalo, Sheep, Goat) |
| Diseases | 18-25 confirmed |
| Symptoms | 15 binary features |
| Multi-label Cases | ~25% |

### 8.2 Species Distribution

| Species | Count | Percentage |
|---------|-------|------------|
| Cattle | ~400 | 38.1% |
| Buffalo | ~300 | 28.6% |
| Sheep | ~200 | 19.0% |
| Goat | ~150 | 14.3% |

### 8.3 Disease Categories

- **Viral (6):** LSD, FMD, HS, PPR, Poxvirus, Other
- **Bacterial (5):** Brucellosis, TB, Mastitis, Metritis, Fascioliasis
- **Parasitic (4):** Tick-borne, Internal parasites, Coccidiosis, Anemia
- **Metabolic (4):** Hypocalcemia, Anorexia, Black Quarter, Enterotoxemia

### 8.4 15 Clinical Symptoms

Fever, Fluid leakage, Diarrhea, Cough, Blisters, Lameness, Stiffening, Nasal discharge, Severe cough, Stomach pain, Bloody stool, Blood in milk, Vaginal signs, Teat abnormalities, Weakness

---

## 9. Expected Results

### 9.1 Performance Comparison

| Model | Macro F1 | Rare F1 | Lead Time |
|-------|----------|---------|-----------|
| Logistic Regression | 0.68 | 0.42 | 0 days |
| XGBoost | 0.81 | 0.55 | 0 days |
| FCNN | 0.84 | 0.58 | 0 days |
| **VetLLM** | **0.89** | **0.77** | **3-7 days** |

### 9.2 Ablation Study

| Configuration | Macro F1 | Impact |
|---------------|----------|--------|
| Full VetLLM | 0.89 | — |
| − Weighted Loss | 0.86 | −3.4% |
| − LSTM Temporal | 0.85 | −4.5% |
| − Multi-Task | 0.86 | −3.4% |

### 9.3 Early Warning

| Disease | Lead Time | Detection Rate |
|---------|-----------|----------------|
| FMD | 3.2 days | 78% |
| LSD | 4.5 days | 82% |
| Brucellosis | 5.8 days | 71% |
| **Average** | **3.9 days** | **76%** |

---

## 10. Research Contributions

### 10.1 Dataset Contribution
First systematic multi-disease Pakistani livestock dataset (1,050 animals, 4 species, 18-25 diseases)

### 10.2 Methodological Contributions
1. **Weighted Multi-Label Loss:** 40% rare disease improvement
2. **LSTM Temporal Modeling:** 3-7 day early warning
3. **Multi-Task Species Learning:** 15-25% data efficiency gain

### 10.3 Clinical Contribution
SHAP interpretability with >80% veterinarian agreement

### 10.4 Societal Impact
\$500M+ potential annual value at national scale

---

## 11. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| 1 | Months 1-2 | Data preprocessing, baselines |
| 2 | Months 3-4 | LSTM, weighted loss implementation |
| 3 | Months 5-6 | Multi-task learning, integration |
| 4 | Months 7-8 | Interpretability, validation |
| 5 | Month 9 | Thesis writing, defense |

---

## 12. Limitations and Mitigation

| Limitation | Mitigation |
|------------|------------|
| Dataset size (1,050) | Cross-validation; federation planned |
| Geographic specificity | Province-specific adaptation roadmap |
| Temporal incompleteness | Masking strategy; static fallback |
| Species imbalance | Multi-task learning compensation |

---

## 13. Conclusion

VetLLM addresses a critical gap in Pakistani veterinary medicine, transforming reactive disease management to proactive early warning through:

- **Authentic regional data** (first Pakistani livestock dataset)
- **Novel methodology** (weighted loss, LSTM, multi-task learning)
- **Significant improvements** (9.9% overall, 40% rare disease, 3-7 day early warning)
- **Clinical usability** (interpretable, deployable on consumer hardware)

**The technology is ready. The need is urgent. The potential impact is transformative.**

---

## 14. References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*, 785-794.
3. Lin, T.Y., et al. (2017). Focal Loss for Dense Object Detection. *IEEE ICCV*, 2980-2988.
4. Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. Rajkomar, A., et al. (2019). Machine Learning in Medicine. *NEJM*, 380, 1347-1358.
6. Saqib, M., et al. (2024). Lumpy Skin Disease Diagnosis Using MobileNetV2. *Computers and Electronics in Agriculture*.
7. Zhang, M.L., & Zhou, Z.H. (2014). Multi-Label Learning Algorithms. *IEEE TKDE*, 26(8), 1819-1837.
8. Caruana, R. (1997). Multitask Learning. *Machine Learning*, 28, 41-75.
9. Pakistan Bureau of Statistics. (2023). Pakistan Economic Survey 2022-23.
10. Knight-Jones, T.J.D., & Rushton, J. (2013). Economic Impacts of FMD. *Preventive Veterinary Medicine*.
11. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
12. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL-HLT*.
13. Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
14. Tjoa, E., & Guan, C. (2021). Survey on Explainable AI. *IEEE TNNLS*, 32(11), 4793-4813.
15. Borisov, V., et al. (2022). Deep Neural Networks and Tabular Data. *IEEE TNNLS*.

---

**Document Prepared:** December 2024  
**Status:** Ready for Defense Proposal Submission

*"From reactive diagnosis to proactive protection—enabling early warning for Pakistan's livestock."*
