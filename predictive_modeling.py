"""
Beauty & Wellness Customer Sentiment & Feedback Analysis
Week 5: Predictive Modeling & Algorithm Selection Architecture
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_and_evaluate_models(df):
    """
    Trains and compares two predictive models:
    1. Regularized Logistic Regression (Interpretable Linear Baseline)
    2. Random Forest Classifier (Non-linear Ensemble Bagging)
    
    Includes:
    - Balanced class weighting for severe class imbalance (positivity skew)
    - Stratified K-Fold cross-validation
    - Comprehensive metric evaluation (Recall, Macro F1, ROC-AUC)
    """
    print("=" * 70)
    print("Executing Beauty & Wellness Predictive Modeling Pipeline (Week 5)")
    print("=" * 70)
    
    df = df.copy()

    # 1. Target Label Creation: Binary Classification
    # 1 = Satisfied (Rating >= 4), 0 = Defect / Grievance (Rating <= 3)
    df['target'] = df['rating'].apply(lambda r: 1 if r >= 4 else 0)
    
    class_counts = df['target'].value_counts()
    print(f"[INFO] Target Distribution: Satisfied={class_counts.get(1, 0)}, Defect/Dissatisfied={class_counts.get(0, 0)}")

    # 2. Vectorization via TF-IDF (Unigrams + Bigrams)
    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df['cleaned_text'])
    y = df['target'].values

    # 3. Stratified Train-Test Split (80/20 preserving minority class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Data Split: Training Samples={X_train.shape[0]}, Testing Samples={X_test.shape[0]}")

    # 4. Model 1: Regularized Logistic Regression with Cost-Sensitive Weighting
    print("
--- Training Model 1: L2-Regularized Logistic Regression ---")
    lr_model = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', max_iter=500, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

    # 5. Model 2: Balanced Random Forest Classifier
    print("--- Training Model 2: Balanced Random Forest Classifier ---")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=4, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    # 6. Evaluation & Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=skf, scoring='f1_macro')
    
    print("
" + "=" * 50)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Logistic Regression 5-Fold Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Logistic Regression Test ROC-AUC:    {roc_auc_score(y_test, y_prob_lr):.4f}")
    print(f"Random Forest Test ROC-AUC:          {roc_auc_score(y_test, y_prob_rf):.4f}")
    
    print("
[CLASSIFICATION REPORT - LOGISTIC REGRESSION (TEST SET)]")
    print(classification_report(y_test, y_pred_lr, target_names=['Defect / Negative', 'Satisfied']))
    
    print("[CONFUSION MATRIX - LOGISTIC REGRESSION]")
    print(confusion_matrix(y_test, y_pred_lr))
    
    return lr_model, rf_model, tfidf

def run_simulation():
    # Synthetic representative corpus spanning multiple classes for pipeline verification
    sample_data = [
        {"rating": 1.0, "cleaned_text": "terrible cystic acne along jawline within days burning redness"},
        {"rating": 1.0, "cleaned_text": "severe allergic reaction red rash chemical burn sensation"},
        {"rating": 2.0, "cleaned_text": "pump dispenser broke second use bottle leaking mess"},
        {"rating": 2.0, "cleaned_text": "clogged pores oily greasy residue artificial perfume odor"},
        {"rating": 3.0, "cleaned_text": "average cleanser basic foaming nothing special dried cheeks slightly"},
        {"rating": 3.0, "cleaned_text": "decent hydration smells heavy price excessive efficacy"},
        {"rating": 4.0, "cleaned_text": "gentle daily face wash nice texture worth money"},
        {"rating": 4.0, "cleaned_text": "solid moisturizer absorbs well packaging slightly difficult"},
        {"rating": 5.0, "cleaned_text": "holy grail moisturizer absorbs instantly repairs skin barrier"},
        {"rating": 5.0, "cleaned_text": "best barrier cream repurchased multiple times highly recommend"},
        {"rating": 5.0, "cleaned_text": "cleared texture glowing skin lightweight non-greasy perfect formula"},
        {"rating": 5.0, "cleaned_text": "miracle serum fine lines smoothed hydrating plumping amazing"}
    ]
    df = pd.DataFrame(sample_data)
    train_and_evaluate_models(df)

if __name__ == "__main__":
    run_simulation()
