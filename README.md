# Mobile Wallet Fraud Detection

## Project Overview
This project uses Machine Learning to detect potentially fraudulent mobile wallet transactions.

The goal is to help financial companies identify suspicious transactions and support better fraud investigation decisions.

## Problem Statement
Fraudulent transactions represent a very small percentage of all mobile wallet transactions, making fraud detection a highly imbalanced classification problem.

This project builds a Machine Learning model to identify suspicious transactions using transaction amount, transaction type, and transaction step.

## Dataset
- **Source:** PaySim Synthetic Financial Dataset
- **Total Transactions:** 6,362,620
- **Features Used:** `step`, `amount`, `type`
- **Target:** `isFraud`

The dataset is synthetic and does not contain real customer transactions.

## Machine Learning Workflow
1. Data Understanding and Exploration (EDA)
2. Data Preprocessing
3. Feature Selection
4. Model Training and Comparison
5. Hyperparameter Tuning
6. Model Evaluation
7. Error Analysis
8. Streamlit Web Application

## Models
The following models were evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting

**Final Model:** Random Forest

## Results

The final model was evaluated on an untouched test set.

| Metric | Result |
|---|---:|
| Average Precision | 0.2676 |
| Precision | 0.3029 |
| Recall | 0.2752 |
| F1-score | 0.2884 |

**Decision Threshold:** 0.95

The threshold was selected using the validation set to maximize the F1-score.

## Error Analysis
- False Positives: 780
- False Negatives: 893

Most false positives were TRANSFER transactions
## Web Application

The project includes a live Streamlit application.

**Live Demo:** https://mobile-wallet-fraud-detection-dnvv4tz4ivaxanak6h9cls.streamlit.app/

The application supports:
- Single transaction prediction.
- Batch prediction using CSV files.
- Fraud probability display.
- Downloading prediction results.

## Project Screenshots

Screenshots of the application interface and prediction results are available in the `screenshots` folder.
