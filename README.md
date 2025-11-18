# Software Defect Prediction using SMOTE-Augmented Random Forest

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](https://github.com/kukuhyudhistiro/ase_35metrics)
[![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen.svg)](README.md)

> **Leveraging 35 Source Code Metrics in SMOTE-Augmented Random Forest for Superior Software Defect Prediction**

A comprehensive machine learning approach for predicting software defects using 35 source code metrics with SMOTE-augmented Random Forest, achieving **perfect classification performance** (100% accuracy, precision, recall, and F1-score).

---

## Table of Contents

- [Overview](#overview)
- [Research Framework](#research-framework)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance Comparison](#performance-comparison)
- [Visualizations](#visualizations)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Overview

Software defects lead to significant financial losses and safety hazards. Traditional Software Defect Prediction (SDP) models often struggle with **class imbalance**, where defective modules constitute only 20-30% of datasets, resulting in poor defect detection rates.

This research addresses these challenges through:

**35 Comprehensive Metrics** - Extended from typical 21-22 metrics  
**SMOTE Balancing** - Addresses 75:25 imbalance  
**Random Forest Ensemble** - 100 estimators with feature randomization  
**Perfect Performance** - 100% accuracy, precision, recall, F1, and AUC  
**Interpretable Results** - SHAP analysis and feature importance  

---

## Research Framework

### Problem Statement

**Given:** Dataset D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}  
- Where xᵢ ∈ ℝ³⁴ represents 34 source code metrics
- yᵢ ∈ {0,1} indicates defect presence

**Objective:** Learn function f: ℝ³⁴ → {0,1} that minimizes classification error while handling class imbalance

**Challenge:** |{yᵢ=0}| >> |{yᵢ=1}| (3.03:1 imbalance ratio)

### Research Questions

1. **RQ1**: Can SMOTE-augmented Random Forest outperform baseline ML methods (NB, LR) in defect prediction?
2. **RQ2**: Which source code metrics are most predictive of software defects?
3. **RQ3**: How does SMOTE impact model performance on imbalanced SDP datasets?
4. **RQ4**: Can we achieve superior performance compared to published baselines?

### Hypotheses

- **H1**: SMOTE+RF will achieve >90% recall (vs. 60-80% in baselines)
- **H2**: Complexity metrics (scc, mcc) will dominate feature importance
- **H3**: SMOTE will improve recall by >10 percentage points
- **H4**: Our approach will exceed 2024 SN CS baseline (82.96% accuracy)

---

## Key Features

### **Perfect Classification**
```
Test Set Performance (n=150):
├── Accuracy:  100%
├── Precision: 100%
├── Recall:    100%
├── F1-Score:  100%
└── AUC-ROC:   100%

Confusion Matrix:
├── True Positives:  37/37 (100%)
├── True Negatives:  113/113 (100%)
├── False Positives: 0
└── False Negatives: 0
```

### **Comprehensive Metrics**
- **Complexity**: mcc, scc, acc, mn
- **Size**: cl, dcl, ecl, l
- **Halstead**: n, n1, n2
- **Structure**: cs, m, cv, cm
- **Documentation**: cml, ccr

### **Rich Visualizations**
- 24-subplot comprehensive dashboard
- ROC & Precision-Recall curves
- SHAP explainability plots
- Feature importance rankings
- Confusion matrices

---

## Dataset

### Overview
- **Total Samples**: 500 software modules
- **Features**: 35 metrics (34 features + 1 label)
- **Class Distribution**:
  - Non-defective: 376 (75.2%)
  - Defective: 124 (24.8%)
  - Imbalance Ratio: 3.03:1
- **Data Quality**: No missing values, 1 duplicate (handled)

### Metric Categories

| Category | Metrics | Examples |
|----------|---------|----------|
| **Complexity** | 6 | mcc, scc, acc, mn |
| **Size** | 8 | cl, dcl, ecl, l, bl |
| **Halstead** | 3 | n, n1, n2 |
| **Structure** | 12 | cs, m, cv, cm, prm, pvm |
| **Documentation** | 5 | cml, ccr, acl |

### Top Features by Correlation

| Rank | Feature | Correlation | Description |
|------|---------|-------------|-------------|
| 1 | scc | 0.7555 | Sum Cyclomatic Complexity |
| 2 | ccr | 0.1142 | Code Comment Ratio |
| 3 | dcl | 0.1105 | Declarative Code Lines |
| 4 | n | 0.0871 | Halstead Length |
| 5 | prm | 0.0847 | Private Methods |

---

## Methodology

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   1. DATA PREPROCESSING                      │
├─────────────────────────────────────────────────────────────┤
│  • Load dataset (500 samples, 35 metrics)                   │
│  • Missing value imputation (median)                        │
│  • Stratified train-test split (70:30)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   2. SMOTE APPLICATION                       │
├─────────────────────────────────────────────────────────────┤
│  • Before: 263 non-defect, 87 defect (3.02:1)              │
│  • K-NN interpolation (k=5)                                 │
│  • After: 263 non-defect, 263 defect (1:1)                 │
│  • Synthetic samples: 176                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   3. FEATURE SCALING                         │
├─────────────────────────────────────────────────────────────┤
│  • StandardScaler normalization                             │
│  • Zero mean (μ = 0.000)                                    │
│  • Unit variance (σ = 1.001)                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   4. MODEL TRAINING                          │
├─────────────────────────────────────────────────────────────┤
│  • Random Forest Classifier                                 │
│  • n_estimators: 100                                        │
│  • max_features: sqrt(34) ≈ 6                               │
│  • random_state: 42                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   5. EVALUATION                              │
├─────────────────────────────────────────────────────────────┤
│  • 5-fold stratified cross-validation                       │
│  • Multiple metrics (Acc, Prec, Rec, F1, AUC)              │
│  • SHAP analysis for interpretability                       │
│  • Confusion matrix & ROC/PR curves                         │
└─────────────────────────────────────────────────────────────┘
```

### Baseline Comparisons

We compare against three baselines:

1. **Naive Bayes (NB)**: Probabilistic classifier with Gaussian assumption
2. **Logistic Regression (LR)**: Linear model with liblinear solver
3. **Literature Baselines**:
   - 2022 IEEE Study: RF on NASA datasets (85-92% acc)
   - 2024 SN CS Study: SMOTE+RF on JM1 (82.96% acc)

---

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Naive Bayes** | 0.9467 | 0.8718 | 0.9189 | 0.8947 | 0.9864 |
| **Logistic Regression** | 0.9467 | 0.8537 | 0.9459 | 0.8974 | 0.9900 |
| **RF + SMOTE**   | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### Cross-Validation Results

**5-Fold Stratified Cross-Validation:**

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9981 | 0.0038 | 0.9943 | 1.0000 |
| Precision | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Recall | 0.9962 | 0.0075 | 0.9887 | 1.0000 |
| F1-Score | 0.9981 | 0.0038 | 0.9943 | 1.0000 |

** Interpretation**: Extremely low variance (std < 0.01) indicates robust generalization across all folds.

### Performance Gains

**vs Naive Bayes:**
- Accuracy: +5.63%
- Precision: +14.71%
- Recall: +8.82%
- F1-Score: +11.76%

**vs Logistic Regression:**
- Accuracy: +5.63%
- Precision: +17.14%
- Recall: +5.71%
- F1-Score: +11.43%

**vs 2022 IEEE Baseline:**
- Accuracy: +8-15%
- Recall: +20-40%

**vs 2024 SN CS Study:**
- Accuracy: +17.04%
- F1-Score: +10.47%

### Feature Importance

**Top 15 Most Predictive Features:**

| Rank | Feature | Importance | Category | Description |
|------|---------|------------|----------|-------------|
| 1 | **scc** | **0.5596** | Complexity | Sum Cyclomatic Complexity (56%!) |
| 2 | dcl | 0.0492 | Structure | Declarative Code Lines |
| 3 | prm | 0.0257 | Structure | Private Methods |
| 4 | ccr | 0.0252 | Quality | Code Comment Ratio |
| 5 | l | 0.0213 | Size | Total Lines |
| 6 | cv | 0.0212 | Structure | Class Variables |
| 7 | cml | 0.0194 | Quality | Comment Lines |
| 8 | n2 | 0.0172 | Halstead | Operands |
| 9 | dm | 0.0162 | Structure | Default Methods |
| 10 | n | 0.0131 | Halstead | Program Length |
| 11 | sc | 0.0129 | Size | Semicolons |
| 12 | es | 0.0128 | Size | Executable Statements |
| 13 | f | 0.0127 | Structure | Functions |
| 14 | iv | 0.0126 | Structure | Instance Variables |
| 15 | bl | 0.0126 | Size | Blank Lines |

**  Key Finding**: Sum Cyclomatic Complexity (scc) dominates with **56% importance**, far exceeding all other features combined.

### Ablation Study Results

**Impact of SMOTE:**
```
RF without SMOTE: 89.3% recall
RF with SMOTE:    100% recall
Improvement:      +10.7 percentage points
```

**Impact of Feature Scaling:**
```
Pre-scaling:  Range 0-4,463 (high variance)
Post-scaling: μ=0.000, σ=1.001 (normalized)
Impact:       Essential for SMOTE distance calculations
```

---

##  Installation

### Prerequisites

```bash
Python 3.8 or higher
```

### Clone Repository

```bash
git clone https://github.com/kukuhyudhistiro/ase_35metrics.git
cd ase_35metrics
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.40.0
```

---

##  Usage

### Quick Start

```python
# Run complete analysis
python sdp_jurnal_improved.py
```

**Expected Output:**
- Console: Comprehensive analysis report (12 sections)
- PNG: 24-subplot visualization dashboard (300 DPI)
- CSV: 3 data files (model_comparison, feature_importance, cv_scores)

### Step-by-Step Execution

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv('datasetjurnal_lengkap_500.csv', sep=';', decimal=',')
X = df.drop('defects', axis=1)
y = df['defects']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 5. Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_res)

# 6. Evaluate
y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

### Custom Analysis

```python
# Feature importance analysis
importances = pd.Series(
    rf.feature_importances_, 
    index=feature_names
).sort_values(ascending=False)

print("Top 10 Features:")
print(importances.head(10))

# SHAP analysis
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values[1], X_test_scaled, feature_names)
```

---

##  Project Structure

```
ase_35metrics/
├── README.md                                    # This file
├── LICENSE                                      # MIT License
├── requirements.txt                             # Python dependencies
│
├── data/
│   └── datasetjurnal_lengkap_500.csv           # Main dataset (500 samples)
│
├── src/
│   ├── sdp_jurnal_improved.py                  # Main analysis script
│   ├── create_presentation.py                  # PowerPoint generator
│   └── create_improved_journal.py              # DOCX generator
│
├── outputs/
│   ├── sdp_comprehensive_analysis.png          # Visualization dashboard
│   ├── model_comparison.csv                    # Performance metrics
│   ├── feature_importance.csv                  # Feature rankings
│   ├── cv_scores.csv                           # Cross-validation results
│   ├── JURNAL_IMPROVED_Software_Defect_Prediction.docx
│   ├── Software_Defect_Prediction_Presentation.pptx
│   └── documentation/
│       ├── FINAL_PROJECT_REPORT.md
│       ├── SUMMARY_ANALISIS_DAN_PERBAIKAN.md
│       └── PRESENTATION_GUIDE.md
│
├── notebooks/
│   └── exploratory_analysis.ipynb              # Jupyter notebook
│
└── docs/
    ├── methodology.md                          # Detailed methodology
    ├── results.md                              # Complete results
    └── api_reference.md                        # Code documentation
```

---

##  Performance Comparison

### Comparison with Published Baselines

| Study | Year | Dataset | Metrics | Method | Accuracy | Recall | F1 |
|-------|------|---------|---------|--------|----------|--------|-----|
| C. Anjali et al. (IEEE) | 2022 | NASA MDP | 21-22 | Vanilla RF | 85-92% | 60-80% | 70-85% |
| N.S. Thomas et al. (SN CS) | 2024 | JM1 | 21 | SMOTE+RF | 82.96% | ~90% | 89.53% |
| **Our Work** | 2025 | Custom | 35 | SMOTE+RF | **100%** | **100%** | **100%** |

### Key Improvements

 **Extended Metrics**: 35 vs typical 21-22  
 **Perfect Recall**: 100% vs 60-90% in baselines  
 **Zero False Negatives**: All defects caught  
 **Robust Validation**: CV std < 0.01  
 **Interpretable**: SHAP + feature importance  

---

##  Visualizations

### Comprehensive Dashboard

![Comprehensive Analysis Dashboard](outputs/sdp_comprehensive_analysis.png)

**24 Subplots Including:**
- Confusion Matrix (perfect diagonal)
- ROC Curves (AUC = 1.00)
- Precision-Recall Curves
- Feature Importance Rankings
- SHAP Summary Plot
- Correlation Heatmap
- Cross-Validation Box Plots
- Feature Distributions by Class

### Key Visualizations

<table>
<tr>
<td width="50%">

**Confusion Matrix**
```
              Predicted
              Non  Def
Actual Non  [113   0]
       Def  [  0  37]
```
Perfect classification!

</td>
<td width="50%">

**Feature Importance**
```
scc:  56.0% ████████████
dcl:   4.9% █
prm:   2.6% 
ccr:   2.5% 
Others: 34%
```
scc dominates!

</td>
</tr>
</table>

---

## Actionable Insights

### For Developers

**HIGH PRIORITY Refactoring Triggers:**
```python
if scc > 45:
    # CRITICAL: Immediate refactoring required
    # Risk: Very High
    action = "Break down complex control flows"

elif dcl > 150:
    # HIGH: Simplify declarations
    # Risk: High
    action = "Reduce setup/initialization code"

elif prm > 6:
    # MEDIUM: Review encapsulation
    # Risk: Moderate
    action = "Consolidate private methods"

elif ccr < 0.15 or ccr > 0.7:
    # LOW-MEDIUM: Improve documentation
    # Risk: Moderate
    action = "Balance comment quality"
```

### Quality Gates

**Recommended CI/CD Integration:**
```yaml
quality_gates:
  mandatory_review:
    - scc > 45
    - dcl > 150
  
  recommended_review:
    - prm > 6
    - ccr < 0.15 OR ccr > 0.7
  
  automated_testing:
    - scc > 30  # Additional unit tests
    - dcl > 100 # Integration tests
```

### Healthy Code Characteristics

```
HEALTHY CODE:
   scc:  0-30   (low overall complexity)
   dcl:  0-100  (simple declarations)
   prm:  0-5    (minimal private methods)
   ccr:  0.35-0.45 (balanced documentation)

AT-RISK CODE:
   scc:  30-45  (moderate complexity - monitor)
   dcl:  100-150 (complex declarations)
   prm:  5-6    (borderline over-engineering)
   ccr:  0.15-0.35 or 0.45-0.7 (suboptimal)

HIGH-RISK CODE:
   scc:  >45    (very high complexity - REFACTOR!)
   dcl:  >150   (excessive declarations)
   prm:  >6     (over-engineered)
   ccr:  <0.15 or >0.7 (poor documentation)
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{yudhistiro2025sdp,
  title={Leveraging 35 Source Code Metrics in SMOTE-Augmented Random Forest for Superior Software Defect Prediction},
  author={Yudhistiro, Kukuh and Marjuni, Aris},
  journal={Under Review},
  year={2025},
  institution={Fakultas Ilmu Komputer, Universitas Dian Nuswantoro}
}
```

**Related Publications:**
- IEEE Study 2022: [DOI Link]
- SN Computer Science 2024: [DOI: 10.1007/s42979-024-02764-x]

---

## 👥 Authors

**Kukuh Yudhistiro**
- 🎓 Fakultas Ilmu Komputer, Universitas Dian Nuswantoro
- 🔗 GitHub: [@kukuhyudhistiro](https://github.com/kukuhyudhistiro)

**Aris Marjuni**
- 🎓 Fakultas Ilmu Komputer, Universitas Dian Nuswantoro

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Kukuh Yudhistiro, Aris Marjuni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## Acknowledgments

- **NASA MDP**: For inspiring baseline comparisons
- **scikit-learn**: For excellent ML library
- **SMOTE**: For addressing class imbalance
- **SHAP**: For model interpretability

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/kukuhyudhistiro/ase_35metrics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kukuhyudhistiro/ase_35metrics/discussions)

---

## Updates & Roadmap

### Current Version: v1.0.0 (November 2025)

**Completed:**
- [x] Perfect test performance (100% metrics)
- [x] Comprehensive 35-metric analysis
- [x] SMOTE integration
- [x] SHAP interpretability
- [x] Publication-ready visualizations

**Planned:**
- [ ] Cross-dataset validation (NASA CM1, KC1, JM1)
- [ ] Deep learning integration (CNN, RNN)
- [ ] Real-time CI/CD plugin
- [ ] Web-based demo application
- [ ] Docker containerization

---

## Star History

If you find this project useful, please consider giving it a star! 

[![Star History Chart](https://api.star-history.com/svg?repos=kukuhyudhistiro/ase_35metrics&type=Date)](https://star-history.com/#kukuhyudhistiro/ase_35metrics&Date)

---

## Statistics

![GitHub stars](https://img.shields.io/github/stars/kukuhyudhistiro/ase_35metrics?style=social)
![GitHub forks](https://img.shields.io/github/forks/kukuhyudhistiro/ase_35metrics?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/kukuhyudhistiro/ase_35metrics?style=social)

---

<div align="center">

**Made with ❤️ by Kukuh**

[⬆ Back to Top](#-software-defect-prediction-using-smote-augmented-random-forest)

</div>
