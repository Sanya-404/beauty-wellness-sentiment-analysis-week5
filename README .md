# Beauty & Wellness Customer Sentiment & Feedback Analysis

[![Internship Track](https://img.shields.io/badge/Internship-Data%20Science%20with%20Python-blue.svg)](https://github.com)
[![Deliverable](https://img.shields.io/badge/Deliverable-Week%205%20Task-green.svg)](https://github.com)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)

## 📌 Project Overview
This repository contains the end-to-end data processing, exploratory data analysis, feature engineering, and machine learning architecture for evaluating consumer sentiment, product efficacy, and feedback patterns in the **Beauty and Personal Care E-Commerce sector**.

While aggregate star ratings (1 to 5 stars) provide broad indicators of customer satisfaction, they fail to isolate underlying defect drivers—such as chemical allergies, formulation breakouts, and dispensing pump malfunctions. This project implements an automated, reproducible Natural Language Processing (NLP) framework to extract actionable intelligence from unstructured consumer reviews.

---

## 🎯 Week 5 Objectives: Predictive Modeling & Algorithm Selection
- **Multi-Algorithm Evaluation:** Comparative assessment of **L2-Regularized Logistic Regression**, **Balanced Random Forest**, and **Multinomial Naive Bayes** across sparse text spaces and engineered interaction features.
- **Cost-Sensitive Class Balancing:** Enforcing balanced class weights and threshold tuning to mitigate the extreme 72% positivity bias in cosmetic reviews.
- **Evaluation Metrics Priority:** Prioritizing **Minority Class Recall ($\ge 0.85$)** and **Macro F1-Score** over surface-level Accuracy to prevent dangerous False Negatives in dermal allergy detection.
- **Robust Validation Architecture:** Enforcing **Stratified 5-Fold Cross-Validation** to guarantee model generalizability and avoid data leakage.
- **Executive Business Translation:** Mapping model probabilities to automated cosmetic R&D defect triage, supplier packaging alerts, and proactive churn recovery.

---

## 📂 Repository Structure
```text
├── README.md                    # Project documentation, modeling framework & governance
├── predictive_modeling.py       # Week 5: Multi-model training, evaluation & cross-validation pipeline
├── feature_engineering.py       # Week 4: Multi-dimensional feature engineering & statistical tests
├── data_cleaning_pipeline.py    # Week 3: Production 5-gate cleaning & preprocessing pipeline
├── eda_script.py                # Week 2: Initial exploratory data analysis & N-gram modeling
└── requirements.txt             # Project environment dependencies
```

---

## 🔬 Predictive Algorithms Comparative Summary

| Algorithm | Family & Mechanism | Domain Strengths | Hyperparameter Controls |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear Sigmoid Classification | Highly interpretable coefficients; fast inference; transparent feature attribution. | `C=1.0`, `penalty='l2'`, `class_weight='balanced'` |
| **Random Forest** | Ensemble Decision Trees (Bagging) | Captures non-linear feature interactions (`skin_type` x `price` x `category`) natively. | `n_estimators=100`, `max_depth=12`, `min_samples_leaf=4` |
| **Multinomial Naive Bayes** | Probabilistic Bayes Theorem | Rapid training speed; suitable as lightweight sparse text baseline. | `alpha=1.0` (Additive Laplace smoothing) |

---

## 🚀 Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/avinashsagar02/beauty-wellness-sentiment-analysis.git
cd beauty-wellness-sentiment-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Week 5 Predictive Modeling Pipeline
```bash
python predictive_modeling.py
```

---

## 🛠️ Tech Stack & Tooling Justification
- **Python 3.10+**: Core runtime.
- **Scikit-Learn**: Vectorizers (`TfidfVectorizer`), Classifiers (`LogisticRegression`, `RandomForestClassifier`), Cross-Validation (`StratifiedKFold`), and Metrics (`classification_report`, `roc_auc_score`).
- **Pandas & NumPy**: Data structure management and matrix calculations.
- **SciPy**: Statistical distribution testing and hypothesis confirmation.
