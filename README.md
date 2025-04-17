# Fraud Detection System - README

## Overview
This repository contains a comprehensive analysis of online payments data containing fraudulent and legitimate transactions. I employ multiple machine learning models to detect fraudulent financial transactions and then use the best model to construct risk score for each transaction. The notebook includes data preprocessing, feature engineering, model training with class imbalance handling, and risk scoring capabilities.

## Key Features
- **Multiple Model Evaluation**: Tests Logistic Regression, Random Forest, and Deep Neural Networks
- **Class Imbalance Handling**: Implements SMOTE (Synthetic Minority Over-sampling Technique)
- **Comprehensive Feature Engineering**: Creates meaningful transaction features
- **Risk Scoring**: Generates interpretable risk categories for transactions
- **Detailed Evaluation**: Provides multiple performance metrics and visualizations

## Dataset
For the analysis, I use the Online Payments Fraud Detection Dataset from [Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset/data) containing financial transaction records (total 6362620) with fraud labels. The class "isFraud" is heavily imbalanced with only 0.13% transactions as fraudulunt. Therefore, I use SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your transaction data in the `data/` directory
2. Run the script:
```python
python fraud_risk_scoring.py
```

## Model Pipeline
1. **Data Preprocessing**:
   - Feature engineering (time-based features, transaction ratios, etc.)
   - One-hot encoding of categorical variables
   - Robust scaling of numerical features

2. **Model Training**:
   - Logistic Regression
   - Random Forest
   - Deep Neural Network

3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1 score, and ROC AUC scores
   - Confusion matrices

## Key Findings
- Dataset shows extreme class imbalance (0.13% fraud)
- Random Forest with SMOTE achieved highest recall (99.67%)

## Output Files
- `fraud_scoring_models.pkl`: Saved best model (Random Forest)
- `model_metadata.json`: Training metadata and performance metrics
- `scored_transactions.csv`: All transactions with risk scores
- `1. Top_features.pdf`: Visualization of feature importances
- `2. Risk_score_comparison.pdf`: Risk score comparison visualization


## Performance Considerations
- Neural networks are skipped for SMOTE as they handle class imbalance internally
- Random Forest uses `balanced_subsample` for built-in class weighting
- Early stopping implemented for neural networks

## Risk Categories
Transactions are classified into 5 risk levels based on predicted fraud probability:
1. Very Low (0-10%)
2. Low (10-30%)
3. Medium (30-70%)
4. High (70-90%)
5. Very High (90-100%)
