# Assignment 4: Telecom Customer Churn Prediction Machine Learning Pipeline

## Overview
An end-to-end supervised machine learning classification pipeline designed to predict telecom customer churn from demographic, billing, and behavioral usage indicators. The pipeline addresses extreme real-world challenges including missing value imputation, multi-type feature encoding, class imbalance, hyperparameter tuning, decision threshold calibration, and model interpretability.

## Files
- `customer_churn_prediction.ipynb` - Complete end-to-end Jupyter notebook covering EDA, preprocessing pipelines, class rebalancing, model benchmarking, and prediction generation.
- `train.csv` - Labeled training dataset containing demographic, contract, and service usage features.
- `test.csv` - Unlabeled evaluation dataset for generalization scoring.
- `predictions.csv` - Final test predictions and predicted churn probabilities.
- `Customer_Churn_Pipeline_Report.pdf` - Comprehensive technical report documenting feature engineering decisions, cross-validation metrics, ROC-AUC comparisons, and business impact.

## Techniques & Methodologies Used
- **Data Remediation & Cleaning:** Imputation of corrupted `TotalCharges` values, type conversions, and zero-leakage data splitting.
- **Scikit-Learn Preprocessing Pipelines:** `ColumnTransformer` integrating `StandardScaler` for numeric features and `OneHotEncoder` for multi-class categorical variables.
- **Class Imbalance Handling:** Evaluation of Synthetic Minority Over-sampling Technique (SMOTE) and algorithmic cost-sensitive class weighting.
- **Model Training & Benchmarking:**
  - Regularized Logistic Regression (L1/L2 penalties)
  - Random Forest Classifier (ensemble bagging with bootstrap aggregation)
  - Gradient Boosted Decision Trees (Hyperparameter tuning via Grid/Randomized Search)
- **Evaluation Metrics:** Stratified K-Fold cross-validation, ROC-AUC score, Precision-Recall AUC, F1-Score, and profit-driven decision threshold optimization.
- **Model Explainability:** Permutation Feature Importance and feature weight extraction to identify primary churn drivers (e.g., contract type, tenure, monthly charges).

## Author
Ahmadreza Jafari
