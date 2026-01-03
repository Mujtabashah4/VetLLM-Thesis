# Research Findings and Analysis: DEEPTAG and VETLLM
## Baseline Methods for Veterinary Diagnosis Coding
### A Comprehensive Review for Thesis Defense



---

## Abstract

This comprehensive research report examines two foundational papers in automated veterinary diagnosis coding: **DEEPTAG (2018)** and **VETLLM (2023)**. These papers represent distinct paradigms in machine learning for clinical natural language processing and establish critical baselines for advancing the field. DEEPTAG pioneered deep learning approaches using bidirectional LSTMs with hierarchical constraints, while VETLLM demonstrates the transformative power of transfer learning from general-purpose large language models. This analysis reveals that VETLLM achieves superior performance with 500x fewer training examples than traditional supervised methods, while DEEPTAG introduces valuable human-machine collaboration mechanisms. The report synthesizes key findings, performance metrics, methodological comparisons, and implications for future thesis research, providing a comprehensive foundation for thesis defense and academic discussion.

---

## Table of Contents

1. [Introduction and Problem Statement](#introduction)
2. [DEEPTAG: Deep Learning with Hierarchical Structure (2018)](#deeptag)
3. [VETLLM: Large Language Models for Veterinary Diagnosis (2023)](#vetllm)
4. [Comparative Analysis: Evolution of Baseline Methods](#comparative)
5. [Critical Findings for Thesis Defense](#critical-findings)
6. [Implications for Future Thesis Research](#implications)
7. [Detailed Technical Comparisons](#technical)
8. [Conclusion](#conclusion)
9. [Performance Tables Summary](#appendix)

---

## Introduction and Problem Statement {#introduction}

### The Veterinary Medicine Coding Challenge

Veterinary medicine faces a critical infrastructure gap that human healthcare addressed decades ago: **systematic diagnostic coding**. While human medical records utilize standardized coding systems such as ICD-9 and ICD-10, veterinary clinical notes remain entirely in free-text format without SNOMED-CT (Systematized Nomenclature of Medicine Clinical Terms) disease codes. This absence of standardized coding creates significant barriers to clinical research, epidemiological tracking, quality improvement initiatives, and translational medicine.

#### Why This Problem Matters

The lack of diagnostic coding in veterinary medicine has profound implications:

- **60-70% of emerging infectious diseases originate in animals**
- Companion animals serve as important spontaneous models for naturally occurring human diseases
- Without systematic diagnostic coding in veterinary records, researchers cannot efficiently:
  - Identify patient cohorts for disease-specific studies
  - Track disease prevalence across institutions and time periods
  - Conduct comparative effectiveness research that could inform human medicine

### Clinical and Research Impact

The problem statement extends beyond mere convenience:

| Impact Area | Description |
|---|---|
| **Research Cohort Identification** | Manual chart review required to identify patients with specific diagnoses, consuming months of researcher time |
| **Disease Surveillance** | Impossible to track disease prevalence changes across veterinary institutions and populations |
| **Translational Medicine** | Blocks comparative research between veterinary and human disease presentations |
| **Quality Improvement** | Prevents data-driven clinical quality initiatives and outcome tracking |
| **Public Health** | Hinders early detection of emerging zoonotic diseases |

### Automated Coding as Solution

Automated methods to extract and code diagnoses from clinical notes offer a potential solution. However, veterinary medicine presents unique challenges compared to human clinical NLP:

- Smaller annotated datasets
- Specialized vocabulary
- Diverse clinical contexts (from emergency referral centers to primary care practices)
- High-dimensional label spaces (4,577+ SNOMED-CT codes across multiple semantic levels)

**DEEPTAG and VETLLM represent two distinct approaches** to solving this problem, developed five years apart and reflecting the rapid evolution of machine learning methodologies.

---

## DEEPTAG: Deep Learning with Hierarchical Structure (2018)

### Overview and Motivation

DEEPTAG, introduced by Nie et al. (2018), represents the first application of deep learning to automated veterinary diagnosis coding. The method directly infers SNOMED-CT diagnostic codes from free-text veterinary clinical notes using a specialized neural network architecture that captures hierarchical relationships between disease codes.

**Key Innovation:** Disease codes are not independent classification targets. Rather, they exist within a hierarchical semantic structure where certain diseases are conceptually similar (grouped under meta-diseases), and this structural information provides valuable inductive bias for learning.

### Architecture and Technical Approach

#### BLSTM Foundation

DEEPTAG employs a **bidirectional LSTM (BLSTM)** neural network as its core architecture. The BLSTM processes clinical notes in both forward and backward directions, allowing the model to capture contextual information from both preceding and following words. This bidirectional approach is particularly valuable for clinical notes where diagnostic information may appear scattered throughout the document.

#### Hierarchical Training Objective

The distinguishing feature of DEEPTAG is its **hierarchical training objective**. Rather than treating the 42 disease codes as independent binary classification targets, the algorithm groups codes into 18 meta-diseases based on SNOMED-CT taxonomy relationships.

**Example Meta-Disease Grouping:**
- Various types of neoplasms are grouped together as a meta-disease
- Various gastrointestinal disorders are grouped together

The hierarchical constraint uses an **L₂-based distance objective** that:
1. Encourages embeddings of codes within the same meta-disease to be close together
2. Pushes different meta-disease centroids apart
3. Provides explicit regularization that helps the model learn more generalizable representations

#### DeepTag-M Variant

The paper introduces **DeepTag-M**, which augments the basic architecture by adding meta-disease prediction as an auxiliary task. Meta-diseases serve as intermediate supervision signals, guiding the model to first classify into broader disease categories before predicting specific codes.

### Training Data and Dataset Composition

#### CSU Dataset

DEEPTAG was trained on **112,558 expert-annotated veterinary notes** from Colorado State University (CSU) College of Veterinary Medicine, a tertiary referral center with specialized oncology, orthopedics, and internal medicine services.

**Dataset Characteristics:**
- Each note manually annotated with multiple SNOMED-CT codes by veterinary clinical experts
- Average of **8 codes per note**
- **Important Bias:** Cancer-related diagnoses represent approximately 30% of the dataset due to CSU's specialized cancer center

#### Cross-Hospital Evaluation

To assess generalization, researchers evaluated performance on an external validation set from a private practice veterinary clinic in Northern California (PP dataset) containing **586 notes**. This cross-hospital evaluation is methodologically important because it assesses whether models trained at a tertiary referral center can transfer to typical primary and secondary care practices.

### Performance Results

#### In-Distribution Performance (CSU)

On the held-out test set from CSU (5,628 notes), DEEPTAG achieved:

| Metric | Value | Interpretation |
|---|---|---|
| **Weighted F1 Score** | 0.68 | Primary accuracy metric across all codes |
| **Exact Match Ratio** | 48.4% | Notes with perfectly predicted diagnoses |
| **Unweighted Precision** | 79.9% | False positive rate control |
| **Unweighted Recall** | 62.1% | True positive rate (sensitivity) |

The weighted F1 of 0.68 represents the harmonic mean of precision and recall, weighted by code frequency. This reflects practical prediction accuracy where more common diagnoses are weighted appropriately.

#### Cross-Hospital Performance (PP)

Performance degraded substantially when applied to out-of-distribution private practice data:

| Metric | CSU Value | PP Value | Domain Drop |
|---|---|---|---|
| **Weighted F1 Score** | 0.68 | 0.432 | **-23.8 points** |
| **Exact Match Ratio** | 48.4% | 17.4% | **-31.0 points** |

**Critical Finding:** The 23.8-point F1 drop from in-distribution to out-of-distribution evaluation represents a critical limitation and motivated significant subsequent research into domain adaptation.

#### Disease-Specific Performance Analysis

**Critical Insights:** Regression analysis controlling for training data size revealed that the **number of disease subtypes** (indicator of label diversity) **significantly predicted lower F1 scores (p < 0.001)**.

**Examples:**

| Disease | Training Examples | Subtypes | F1 Score | Pattern |
|---|---|---|---|---|
| **Disorder of digestive system** | 22,589 | 694 | 0.715 | Many subtypes → Lower F1 |
| **Disorder of hematopoietic cell proliferation** | 7,294 | 22 | 0.91 | Few subtypes → High F1 |

**Counterintuitive Finding:** Despite having 3× more training examples, the digestive system disorder achieved lower performance due to its 31× higher subtype diversity.

#### Hierarchical Objective Impact

Ablation analysis demonstrated the value of the hierarchical constraint. **DeepTag outperformed baseline BLSTM by 4.5 percentage points** on the cross-hospital PP dataset (43.2% vs. 36.9% F1), providing quantitative evidence that hierarchical structural constraints improve generalization.

### Technical Innovation: Learning to Abstain

#### Motivation for Human-Machine Collaboration

A distinctive contribution of DEEPTAG is the **learning to abstain mechanism**, which addresses a critical practical consideration: **not all predictions should be automated**. For borderline cases where the model has low confidence, human experts should make the final determination. This enables a human-in-the-loop workflow where the system handles routine cases while flagging uncertain cases for expert review.

#### Implementation

DeepTag-abstain learns a nonlinear function to assess multi-label prediction confidence beyond simple logistic output thresholding. Rather than using raw sigmoid outputs as confidence scores, the model learns a more sophisticated confidence function that considers the relationships between predicted codes and the specific clinical context.

#### Clinical Impact

When progressively removing 20-30% of the most uncertain predictions for human review, the system demonstrates **steeper performance improvements** compared to confidence-threshold baselines. This means that investing human expertise into the model's most uncertain cases yields the highest return on human effort.

### Analysis of Cross-Hospital Generalization Gap

#### Root Causes of Domain Shift

The substantial performance drop from CSU (0.68) to PP (0.432) results from two primary mechanisms:

##### Style and Vocabulary Mismatch

PP notes are systematically different from CSU notes:

- **Length:** PP notes average 253 words vs. 368 words in CSU (31% shorter)
- **Style:** More abbreviations, informal clinical shorthand, fewer structured sections
- **Vocabulary:** After filtering numbers, **15.4% of PP words do not appear in CSU training vocabulary**

##### Disease Distribution Shift

The disease mix differs fundamentally:

- **CSU Specialization:** 30% cancer-related diagnoses due to specialized oncology center
- **PP Mix:** Typical primary/secondary care with different disease prevalence
- **Example:** "Neoplasm and/or hamartoma" has 749 distinct subtypes in CSU but only 7 in PP

#### Implications for Deployment

**Critical Principle:** Models trained on tertiary referral center data do not automatically transfer to primary care practices. Clinical deployment requires either:

1. Extensive cross-hospital training data
2. Domain adaptation techniques
3. Retraining at deployment sites

---

## VETLLM: Large Language Models for Veterinary Diagnosis (2023) {#vetllm}

### Overview and Paradigm Shift

VETLLM, introduced by Jiang et al. (2023), represents a **fundamental shift** from DEEPTAG's specialized supervised learning approach. Rather than building a veterinary-specific neural network trained from scratch, VETLLM leverages pre-trained large language models (LLMs) with parameter-efficient fine-tuning.

**Broader Context:** This approach represents a broader trend in machine learning: the recognition that general-purpose foundation models trained on massive text corpora contain substantial generalizable knowledge that can be adapted to specialized domains with minimal training data.

### Architecture and Technical Approach

#### Base Model Selection

VETLLM builds upon **Alpaca-7B**, an open-source instruction-tuned large language model with 7 billion parameters. Alpaca is derived from Meta's LLaMA model and fine-tuned on instruction-following examples, making it responsive to natural language instructions rather than just raw text completion.

**Practical Advantage:** The choice of Alpaca-7B reflects practical constraints—a 7B parameter model fits on consumer-grade GPUs (8GB VRAM), unlike larger proprietary models that require enterprise infrastructure.

#### Low-Rank Adaptation (LoRA)

Rather than full fine-tuning (which would update all 7 billion parameters), VETLLM employs **Low-Rank Adaptation (LoRA)**, a parameter-efficient technique. LoRA freezes the original model weights and introduces trainable low-rank matrices into attention layers.

**Mathematical Formulation:**
- W' = W₀ + ΔW = W₀ + BA
- Where W₀ is the original weight matrix
- B and A are low-rank matrices (e.g., rank=8)
- BA represents a small perturbation to the original weights

**Impact:** Reduces trainable parameters by orders of magnitude while maintaining model capacity.

#### Configuration Details

The fine-tuning configuration for VETLLM:

- **Base Model:** Alpaca-7B (7 billion parameters)
- **Fine-tuning Data:** 5,000 notes from CSU training split
- **LoRA Rank:** 8
- **LoRA Alpha:** 16
- **Target Modules:** Query projections (q_proj) and value projections (v_proj) in attention layers
- **Hardware:** 4 NVIDIA A4000 GPUs
- **Training Duration:** 48 hours

#### Task Formulation

VETLLM formulates diagnosis prediction as a natural language task rather than structured machine learning. For each disease code, the model is queried: **"Does this note mention [disease]?"** with answers structured as "Yes" or "No." 

This one-shot-per-disease formulation simplifies the resolver design but requires sequential queries (one per disease code).

### Performance Metrics and Results

#### In-Distribution Performance (CSU)

On CSU data, VETLLM dramatically outperformed previous methods:

| Metric | VETLLM | VetTag | Improvement | Type |
|---|---|---|---|---|
| **F1 Score** | 0.747 | 0.592 | +0.155 | +26.2% |
| **Exact Match Ratio** | 53.5% | 49.3% | +4.2 pts | +8.5% |
| **Precision** | 0.726 | — | — | — |
| **Recall** | 0.774 | — | — | — |

The **9.8 percentage point F1 improvement** is substantial in clinical NLP where each point represents meaningful improvement in practical accuracy.

#### Cross-Hospital Performance (PP)

**Most Impressively**, VETLLM achieved superior performance on out-of-distribution data:

| Metric | VETLLM | VetTag | Improvement | Relative Gain |
|---|---|---|---|---|
| **F1 Score** | 0.637 | 0.422 | +0.215 | **+50.7%** |
| **Exact Match Ratio** | 38.0% | 30.1% | +7.9 pts | +26.2% |
| **Precision** | 0.661 | — | — | — |
| **Recall** | 0.630 | — | — | — |

The **50.7% relative improvement on cross-hospital data** is exceptional, suggesting that general-purpose LLM pretraining provides superior domain robustness compared to supervised approaches trained exclusively on in-domain data.

#### Domain Robustness Analysis

**Comparing domain drops:**

- **DEEPTAG:** F1 drop of 23.8 points (0.68 → 0.432), or -35% relative
- **VETLLM:** F1 drop of 11 points (0.747 → 0.637), or -14.7% relative

**Critical Insight:** VETLLM exhibits substantially better domain robustness, suggesting that the broad medical knowledge learned during LLM pretraining transfers more effectively to new institutional contexts.

### Zero-Shot Capability

A **remarkable finding** is that even without veterinary fine-tuning, Alpaca-7B demonstrated non-trivial zero-shot performance:

| Setting | F1 Score | Comparison | Gap |
|---|---|---|---|
| **Zero-shot CSU** | 0.538 | vs. VetTag supervised (0.592) | -6% |
| **Zero-shot PP** | 0.389 | vs. VetTag supervised (0.422) | -7.8% |

The near-parity with VetTag on CSU (only 6% gap) is striking because it requires **zero veterinary-specific training**. This demonstrates that modern LLMs possess substantial medical knowledge from pretraining on internet-scale text corpora.

### Data Efficiency: The Breakthrough Finding

#### Systematic Evaluation

VETLLM systematically evaluated performance with progressively smaller fine-tuning datasets:

| Fine-Tuning Samples | Achieved F1 | Comparison Baseline | Assessment |
|---|---|---|---|
| **200 notes** | ~0.60 | VetTag (0.592) | **Exceeds supervised** |
| **500 notes** | ~0.67 | — | Substantially better |
| **1,000 notes** | ~0.71 | — | Approaching full model |
| **2,000 notes** | ~0.73 | — | High-confidence level |
| **5,000 notes** | 0.747 | Full fine-tuned model | Complete |

#### Implications for Resource-Constrained Settings

This finding has **profound implications**. VetTag required 100,000+ labeled notes to achieve 0.592 F1. VETLLM **exceeds this performance with just 200 fine-tuned notes**—a **500x reduction in annotation burden**.

**For veterinary practices and research institutions with limited resources:**

| Aspect | Traditional Supervised | VETLLM |
|---|---|---|
| **Notes Required** | 100,000+ | 200-500 |
| **Annotation Time** | 6-12 months | 1-2 weeks |
| **Annotation Effort** | Multiple annotators per note | Single clinician |
| **Deployment Timeline** | Months | Weeks |

### Computational Requirements

#### Training Costs

| Resource | Requirement | Implications |
|---|---|---|
| **GPUs Required** | 4× NVIDIA A4000 | ~$8,000 hardware cost |
| **Training Duration** | 48 hours | Can be completed over weekend |
| **Memory per GPU** | ~20 GB total | Feasible on high-end consumer GPUs |

For context, NVIDIA A4000 GPUs cost approximately $2,000 each, placing the hardware investment within reach of most research institutions.

#### Inference Performance

| Operation | Time | Notes |
|---|---|---|
| **Model Loading** | ~15 seconds | One-time cost |
| **Single Disease Query** | ~0.3 seconds | Sequential, per-disease |
| **Full Prediction (9 diseases)** | ~3 seconds | Slower than single-pass models |
| **Batch Processing (1,000 notes)** | ~50 minutes | Reasonable for overnight batch |

The 3-second inference per note is slower than DEEPTAG's single-pass inference but acceptable for non-real-time applications like batch diagnostic coding.

### Sensitivity to Prompt Design

#### Unexpected Brittleness

VETLLM revealed an important consideration: **performance is surprisingly sensitive to subtle prompt variations**. The paper reports that:

- Problem phrasing affects performance
- Information ordering in prompts influences outputs
- Even **trailing whitespace affects predictions**

**Example:** Adding trailing spaces at the beginning of each line measurably changes performance, suggesting that LLMs are sensitive to superficial formatting details.

#### Implications for Deployment

This brittleness creates practical challenges:

- Requires careful prompt engineering before deployment
- May necessitate prompt validation on local data
- Raises questions about model stability under minor input variations
- The authors advocate for systematic research on LLM prompt sensitivity and AI alignment in medical applications

---

## Comparative Analysis: Evolution of Baseline Methods 

### Methodological Timeline

Three distinct papers mark the evolution of veterinary diagnosis coding:

#### DEEPTAG (2018): Pioneering Deep Learning

DEEPTAG introduced deep learning to this problem using:

- Specialized BLSTM architecture
- Hierarchical regularization via code grouping
- Learning-to-abstain human collaboration mechanism
- Coverage: 42 top-level disease codes

**Strengths:**
- First proof-of-concept
- Human-in-the-loop design
- Interpretable via attention mechanisms

**Limitations:**
- Significant cross-hospital domain gap (23.8-point F1 drop)
- Requires extensive labeled data

#### VetTag (2019): Scaling with Pretraining

VetTag advanced the field using:

- Transformer-based architecture
- Large-scale unsupervised pretraining on 1M+ unlabeled notes
- Hierarchical loss across 5 SNOMED-CT depth levels
- Coverage: All 4,577 SNOMED codes

**Strengths:**
- Comprehensive label space
- Better domain robustness than DEEPTAG
- Hierarchical structure

**Limitations:**
- Requires massive labeled datasets
- Computationally expensive
- Months of training

#### VETLLM (2023): Efficient Transfer Learning

VETLLM represents the modern paradigm:

- General-purpose LLM fine-tuning (Alpaca-7B)
- Parameter-efficient adaptation via LoRA
- Minimal labeled data requirement (200+ notes)
- Coverage: 9 high-priority disease classes (limited scope in study)

**Strengths:**
- Exceptional data efficiency
- Superior cross-hospital robustness
- Rapid deployment
- Zero-shot capability

**Limitations:**
- Sequential inference (slower)
- Prompt sensitivity
- Limited to 9 diseases in study
- Less sophisticated human collaboration

### Head-to-Head Performance Comparison

| Dimension | DEEPTAG | VETLLM | Winner | Magnitude |
|---|---|---|---|---|
| **CSU F1 Score** | 0.68 | 0.747 | VETLLM | +9.8% absolute |
| **PP Cross-Hospital F1** | 0.432 | 0.637 | VETLLM | +47.4% relative |
| **Domain Robustness (drop)** | -23.8 pts | -11 pts (rel.) | VETLLM | 2.3× better |
| **Training Data Needed** | 112,558 notes | 5,000 notes | VETLLM | 22.5× less |
| **Notes to Beat Baseline** | — | 200 notes | VETLLM | 500× less |
| **Training Duration** | Weeks-months | 48 hours | VETLLM | 50-100× faster |
| **Hardware Cost** | ~$10,000 | ~$8,000 | Comparable | — |
| **Inference Speed (per note)** | <100 ms | ~3 seconds | DEEPTAG | 30× faster |
| **Abstention Mechanism** | Learning-to-abstain | Basic thresholding | DEEPTAG | More sophisticated |
| **Confidence Assessment** | Multi-label aware | Logit-based | DEEPTAG | Superior |
| **Saliency Maps** | Available | Limited | DEEPTAG | Direct explanation |
| **Code Relationships** | Explicit hierarchy | Implicit in LLM | DEEPTAG | More transparent |
| **Disease Coverage** | 42 top-level | 9 in study | DEEPTAG | Broader |

---

## Critical Findings for Thesis Defense {#critical-findings}

### Finding 1: Domain Generalization Exceeds Dataset Scale in Importance

#### The Core Insight

Both papers identify domain shift as the primary limiting factor in automated veterinary coding, but **VETLLM's superior cross-hospital performance (0.637 vs. DEEPTAG's 0.432) reveals a critical principle:**

> **Leveraging general-purpose pretrained models provides better regularization against domain shift than building specialized architectures trained on large in-domain data.**

#### Evidence

- **DEEPTAG:** Trained on 112,558 CSU notes → F1 = 0.432 on PP (out-of-domain)
- **VETLLM:** Fine-tuned on 5,000 CSU notes using general-purpose pretraining → F1 = 0.637 on PP
- **Key Difference:** VETLLM benefits from broad medical knowledge acquired during pretraining on diverse internet text

**47.4% relative improvement** demonstrates that domain shift is fundamentally different from dataset size—it's about knowledge transfer, not data volume.

#### Implications for Your Thesis

Rather than pursuing exclusively in-domain supervised learning, prioritize approaches that:

1. Leverage general-purpose foundation models
2. Employ efficient adaptation mechanisms (LoRA, prompt engineering, few-shot learning)
3. Validate on diverse out-of-domain test sets
4. Explicitly measure domain robustness

---

### Finding 2: Data Efficiency Represents a Paradigm Shift

#### The Breakthrough

VETLLM's discovery that **200 fine-tuned notes exceed performance of 100,000+ note supervised training** represents a fundamental shift in methodology. This **500× reduction in annotation burden** transforms the practical feasibility of clinical AI deployment.

#### Quantitative Evidence

| Approach | Training Notes | Achieved F1 | Efficiency Ratio |
|---|---|---|---|
| VetTag Supervised | 100,000+ | 0.592 | Baseline |
| VETLLM Fine-tuned (500 notes) | 500 | 0.67 | 200× improvement |
| VETLLM Fine-tuned (200 notes) | 200 | ~0.60 | **500× improvement** |

#### Real-World Impact

For a typical veterinary practice or research institution:

- **Supervised Approach:** 100,000 notes require 6-12 months of annotation (multiple annotators per note for quality control), substantial institutional commitment
- **Transfer Learning:** 200-500 notes require 1-2 weeks of annotation, feasible for single clinician
- **Deployment Window:** Months vs. weeks

#### Thesis Implications

Future work should explore:

1. Minimal data thresholds for different disease categories
2. Few-shot and zero-shot learning techniques
3. Active learning to prioritize which samples to annotate
4. Domain-specific pretraining to further improve efficiency

---

### Finding 3: Label Complexity is a Fundamental Challenge

#### The Hidden Factor

Both papers identify a surprising pattern: **the number of disease subtypes (label diversity) predicts prediction difficulty independent of training set size.**

#### Evidence from DEEPTAG

Regression analysis controlling for training data size revealed that disease subtype count significantly predicted F1 scores (p < 0.001):

| Disease | Train Examples | Subtypes | F1 Score | Pattern |
|---|---|---|---|---|
| Hematopoietic Proliferation | 7,294 | 22 | 0.91 | Few subtypes → High F1 |
| Digestive System Disorder | 22,589 | 694 | 0.715 | Many subtypes → Lower F1 |

Despite having 3× more training examples, the digestive system disorder achieved lower performance due to its 31× higher subtype diversity.

#### Theoretical Interpretation

This suggests that medical coding involves **inherent complexity from the disease ontology itself**. Diseases with many clinical presentations are fundamentally harder to predict, regardless of training data quantity.

**Underlying mechanisms likely include:**

- **Diverse Clinical Presentations:** Diseases with 694 subtypes manifest in many clinically distinct ways, each potentially expressing different vocabulary in notes
- **Label Ambiguity:** Different clinical features may map to different subtypes, creating ambiguous training signals
- **Hierarchical Complexity:** Multiple levels of specificity increase decision boundary complexity

#### Thesis Implications

Rather than simply scaling training data, focus on:

1. **Structural Constraints:** Hierarchical losses, label grouping, multi-task learning
2. **Capable Architectures:** Models with sufficient capacity for high-dimensional label spaces
3. **Auxiliary Information:** Leverage disease hierarchies, clinical context, comorbidity patterns
4. **Label Optimization:** Investigate whether some label hierarchies are more learnable than others

---

### Finding 4: Complementary Strengths Enable Hybrid Approaches

#### The Complementarity

DEEPTAG and VETLLM have distinct strengths and weaknesses that suggest hybrid approaches:

| Dimension | DEEPTAG Strength | VETLLM Strength | Opportunity |
|---|---|---|---|
| Accuracy | Lower | Higher ✓ | Use VETLLM for predictions |
| Inference Speed | Higher ✓ | Lower | Use DEEPTAG for real-time |
| Human Collaboration | Sophisticated ✓ | Basic | Use DEEPTAG for confidence |
| Interpretability | Better ✓ | Limited | Combine for explainability |
| Cross-Hospital | Weaker | Stronger ✓ | Use VETLLM foundation |

#### Proposed Hybrid Architecture

A superior system might combine both approaches:

**Stage 1:** Use VETLLM's superior accuracy for primary predictions (0.637 F1 on cross-hospital data)

**Stage 2:** Apply DEEPTAG's learning-to-abstain mechanism to assess confidence (which predictions warrant human review)

**Stage 3:** For uncertain cases (e.g., bottom 20%), route to human clinician expert review

**Stage 4:** Use DEEPTAG-style attention mechanisms to highlight which text portions informed predictions (explainability)

#### Integration Benefits

This would leverage:

- VETLLM's superior accuracy and domain robustness
- DEEPTAG's human-collaboration mechanism
- DEEPTAG's interpretability features

---

### Finding 5: Rigorous Evaluation Methodology is Essential

#### Gold-Standard Practices

Both papers establish methodological standards for medical NLP evaluation:

1. **Cross-Hospital Testing:** Evaluate on external institutions, not just train/test splits from the same data source
2. **Multiple Metrics:** Report F1, precision, recall, and exact match separately
3. **Stratified Analysis:** Evaluate performance per disease code to reveal hidden gaps
4. **Domain Shift Quantification:** Explicitly measure performance degradation across institutional contexts
5. **Ablation Studies:** Demonstrate the value of key components

#### Why This Matters

A model achieving 0.68 F1 on in-distribution CSU data might seem reasonable. However, DEEPTAG's cross-hospital evaluation revealed the harsh reality: **0.432 F1 on real-world data**. This 23.8-point gap would be **invisible without external validation**.

#### Thesis Implications

Your evaluation protocol should include:

- Primary test set from same institution (establishes baseline ceiling)
- Secondary test sets from 2-3 other institutions (assess generalization)
- Analysis of failure modes on out-of-domain data
- Disease-specific performance breakdowns
- Comparison to prior baselines on the same test sets

---


---

## Detailed Technical Comparisons

### Architecture Comparison

#### DEEPTAG Architecture Flow

```
Clinical Note Text
    ↓
Word Embeddings
    ↓
Bidirectional LSTM (Forward & Backward)
    ↓
Context Vectors per Word
    ↓
Attention Mechanism
    ↓
Document Representation
    ↓
Output Layer with 42 Disease Codes
    ↓
Multi-Label Sigmoid (Independent Predictions)
    ↓ (with Hierarchical Regularization)
Final Predictions with Abstention Option
```

**Key Architectural Features:**

- Word-level processing (rather than subword tokens)
- Bidirectional context synthesis
- Explicit attention to important text portions
- Output-layer embeddings with distance-based constraints
- Separate confidence estimation for abstention

#### VETLLM Architecture Flow

```
Clinical Note + Prompt
    ↓
Tokenization (BPE)
    ↓
Pre-trained LLM Embeddings
    ↓
24 Transformer Layers (Frozen)
    ↓
LoRA Adaptation Layers (Trainable)
    ↓
Attention Layers with Rank-8 Modifications
    ↓
Final Hidden States
    ↓
Output Projection
    ↓
Token Predictions
    ↓
Parse "Yes"/"No" Response
    ↓
Boolean Diagnosis Prediction (Per Disease Query)
```

**Key Architectural Features:**

- Subword tokenization (handles diverse vocabulary)
- Pre-trained transformer foundation
- Parameter-efficient LoRA modifications
- Natural language I/O (prompts and text responses)
- Query-based sequential prediction

### Learning Paradigm Comparison

#### DEEPTAG: Supervised Learning

- **Data:** Requires labeled examples with ground-truth SNOMED codes
- **Objective:** Minimize multi-label classification loss
- **Initialization:** Random weight initialization (no pretraining)
- **Training:** Thousands of optimization steps
- **Regularization:** Hierarchical distance constraints + standard L2 regularization

#### VETLLM: Transfer Learning

- **Data:** Uses fewer labeled examples, benefits from massive general pretraining
- **Objective:** Fine-tune pretrained LLM on veterinary task
- **Initialization:** Pre-trained weights from Alpaca-7B (based on LLaMA)
- **Training:** Relatively few optimization steps (smaller learning rates)
- **Regularization:** Weight decay + LoRA rank constraints

**Key Distinction:** DEEPTAG learns task-specific features from scratch, while VETLLM adapts pre-learned medical knowledge.

---

## Conclusion

### Synthesis of Key Findings

DEEPTAG and VETLLM represent **two complementary approaches** to automated veterinary diagnosis coding, reflecting the evolution of machine learning methodologies:

#### DEEPTAG (2018): Pioneering Specialized Learning

DEEPTAG established the foundation for deep learning in veterinary diagnosis coding. Its key contributions include:

1. **Proof of Concept:** Demonstrated that neural networks could effectively predict SNOMED-CT codes from clinical notes
2. **Hierarchical Constraints:** Showed that disease taxonomy information improves generalization (4.5-point F1 improvement)
3. **Human-Machine Collaboration:** Introduced learning-to-abstain mechanism enabling practical clinical workflows
4. **Cross-Hospital Evaluation:** Established the importance of external validation for assessing generalization

However, **DEEPTAG's 23.8-point F1 drop** from in-distribution to cross-hospital data revealed a critical limitation: specialized models trained on large institutional datasets do not automatically generalize to new clinical contexts.

#### VETLLM (2023): Transfer Learning Revolution

VETLLM demonstrates a fundamentally different paradigm with transformative results:

1. **Superior Accuracy:** Achieves 0.747 F1 (vs. 0.68 for DEEPTAG)
2. **Domain Robustness:** Maintains 0.637 F1 on cross-hospital data (vs. 0.432 for DEEPTAG)—a **47.4% relative improvement**
3. **Data Efficiency:** Requires only 200 fine-tuned notes to exceed supervised baselines trained on 100,000+ notes
4. **Zero-Shot Capability:** Alpaca-7B achieves 0.538 CSU F1 without veterinary training
5. **Rapid Deployment:** Fine-tuning in 48 hours on consumer-grade hardware

**VETLLM's success reveals** that general-purpose foundation models contain sufficient medical knowledge that adaptation to specialized domains requires minimal data.


## Data Preprocessing and SNOMED-CT Mapping for VetLLM Fine-Tuning

### Clinical Data Sources

- **UVAS DLO Cow/Buffalo dataset**: `Verified DLO data - (Cow Buffalo).xlsx` (778 rows).
- **UVAS DLO Sheep/Goat dataset**: `Verified DLO data (Sheep Goat).xlsx` (859 rows).

Each row corresponds to a single clinical case with:
- `Animal Name` (species: Cow, Buffalo, Sheep, Goat),
- a set of **binary symptom columns** (0/1 flags for presence of clinical signs),
- a clinician-assigned **Disease** label.

Rows in which **all symptom columns were zero** (i.e., no positive clinical findings) were excluded because no meaningful clinical note could be constructed from them. This removed 32 cases from the cow/buffalo file and 3 from the sheep/goat file, leaving **746 + 856 = 1,602** clinically informative records used for model training.

### Transformation to Instruction-Tuning Format

We implemented a dedicated preprocessing script (`preprocess_data.py`) to convert these spreadsheets into the **instruction–input–output** format required for VetLLM fine-tuning:

- **Instruction** (fixed across all samples):  
  “Analyze the following veterinary clinical note and predict the SNOMED-CT diagnosis codes.”
- **Input**: a natural-language **clinical note** automatically generated from the binary symptom indicators and animal type, e.g.:  
  “Clinical Note: Cow. Clinical presentation includes epistaxis (nosebleed) and high fever. Physical examination reveals these clinical signs.”
- **Output**: a textual summary of the diagnosis codes, e.g. “Diagnosed conditions: 40214000”.
- **Metadata fields** (not used directly by the model but preserved for auditability): the original `disease` label, `animal` species, and the list of positive `symptoms`.

Symptom column names were normalized into readable medical language using a rule-based mapping; for example, `Continuous Loose Motions` → “persistent diarrhea”, `Blood Leakage From Nose` → “epistaxis (nosebleed)”, and `Mouth Frothing` → “oral frothing”. Only symptoms with value `1` were included in each note, ensuring that every generated description faithfully reflects the underlying binary data.

### SNOMED-CT Mapping and Validation

To avoid arbitrary or synthetic labels, we derived all diagnosis codes from a **manually curated SNOMED-CT report** (`SNOMED_codes.md`) covering the priority livestock diseases of interest. A separate mapping file (`snomed_mapping.json`) links each disease string as it appears in the spreadsheets to one or more **validated SNOMED-CT concept IDs**. Examples include:

- `Anthrax` → **40214000** (Anthrax (disorder)),
- `B.Q` / `Black Quarter` → **29600000** (Blackleg),
- `BABESIOSIS` / `Babesiosis` → **24026003**,
- `Brucellosis` → **75702008**,
- `CCPP` (Contagious caprine pleuropneumonia) → **2260006**,
- `FMD` / `Foot and Mouth` / `Foot and mouth` → **3974005**,
- `H.S` (Hemorrhagic septicemia) → **198462004**,
- `Kataa`, `P.P.R`, `PPR` → **1679004** (Peste des petits ruminants),
- `Liver Fluke` → **4764006** (Fascioliasis),
- `Mastitis` / `Mastits` → **72934000**,
- `Goat Pox` → **57428005**,
- `Theileriosis` → **24694002**,
- `T.B` → **56717001** (Tuberculosis),
- `Rabbies` → **14146002** (Rabies spelling variant in the data).

For diseases that **appear in the Excel files but do not yet have an authoritative SNOMED-CT code** in the report (e.g. `Abortion`, `Foot Rot`, `Ketosis`, `Tympany`), the pipeline preserves the textual disease label in the JSON but leaves the `snomed_codes` field empty. This design guarantees that:

- no synthetic or guessed SNOMED codes are introduced,  
- every code used in training is traceable to a specific vetted concept ID,  
- downstream evaluation can distinguish between **fully coded** and **uncoded** diagnoses.

Overall, the final fine-tuning dataset contains **1,602 real veterinary cases**, each derived directly from UVAS clinical records, transformed into an instruction-following format, and—where possible—annotated with **validated SNOMED-CT diagnosis codes** suitable for robust VetLLM training.


## Final Thoughts

DEEPTAG and VETLLM together provide a comprehensive foundation for understanding the state of veterinary diagnostic NLP. DEEPTAG's pioneering work established the task, demonstrated the importance of hierarchical structure, and revealed the domain adaptation challenge. VETLLM's modern approach proved that general-purpose foundation models can solve this problem more efficiently and robustly.

**Your thesis should acknowledge both papers' contributions while advancing the field by addressing their identified limitations.** Whether through hybrid architectures, improved prompting, enhanced domain adaptation, or better human-machine collaboration, the path forward is clear:

1. **Leverage foundation models**
2. **Prioritize domain robustness**
3. **Minimize annotation requirements**
4. **Respect label structure**
5. **Design for clinical deployment**

---

## Performance Tables Summary 

### Complete Performance Comparison Across All Settings

| Model | Setting | F1 Score | Exact Match | Precision | Recall |
|---|---|---|---|---|
| DEEPTAG | CSU | 0.68 | 48.4% | 79.9% | 62.1% |
| DEEPTAG | PP | 0.432 | 17.4% | — | — |
| VetTag | CSU | 0.592 | 49.3% | — | — |
| VetTag | PP | 0.422 | 30.1% | — | — |
| VETLLM | CSU | 0.747 | 53.5% | 0.726 | 0.774 |
| VETLLM | PP | 0.637 | 38.0% | 0.661 | 0.630 |
| Alpaca-7B Zero-shot | CSU | 0.538 | — | — | — |
| Alpaca-7B Zero-shot | PP | 0.389 | — | — | — |

### Training Dataset Characteristics

| Characteristic | CSU | PP |
|---|---|---|
| Total Notes | 112,558 | 586 |
| Average Words/Note | 368 | 253 |
| Average Codes/Note | 8 | — |
| Primary Specialty | Tertiary Referral | Primary/Secondary Care |
| Cancer Cases | 30% | <5% |

---

**Recommended Citation:** "Research Findings and Analysis: DEEPTAG and VETLLM - Baseline Methods for Veterinary Diagnosis Coding" (2025)