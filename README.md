# Explainable Credit Risk Assessment System

A Streamlit-based machine-learning prototype for exploring loan-default risk and model explainability.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20LightGBM-orange)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-6236FF)](https://shap.readthedocs.io/)

> **Prototype / educational project.** The repository currently uses a small, manually created synthetic dataset (25 rows). Its test set contains only five examples, so reported scores are illustrative and are not reliable estimates of real-world performance. This app must not be used for actual lending decisions.

## Overview

The application accepts applicant and loan details, applies the saved preprocessing pipeline, and uses a trained Support Vector Machine (SVM) model to produce a binary default-risk prediction. The repository also includes scripts for training and comparing a Decision Tree, K-Nearest Neighbors (KNN), LightGBM, and SVM, plus a separate SHAP feature-importance analysis.

## Features

- Interactive Streamlit interface for applicant details
- Numeric scaling and categorical one-hot encoding
- Binary prediction: `0 = no default`, `1 = default`
- Demonstration loan-decision label based on the predicted class
- SVM probability output shown as model confidence (not calibrated probability of repayment)
- Training scripts for SVM, KNN, Decision Tree, and LightGBM
- Test-set metrics including accuracy, precision, recall, F1-score, and confusion matrix
- SHAP-based global feature-importance output and saved plots

## Application Preview

The app is built with Streamlit. Run it locally using the instructions below. A hosted demo link will be added here after deployment is published and verified.

**Live demo:** _Not published yet._

## Input Features

| Feature | Description |
|---|---|
| Age | Applicant age in years |
| Monthly income | Applicant income value used by the sample dataset |
| Loan amount | Requested loan amount |
| Loan term | Duration in months |
| Credit score | Credit-score value |
| Existing debt | Existing debt value |
| Employment years | Years employed |
| Loan purpose | Personal, Home, Business, Education, or Auto |
| Home ownership | Rent, Own, or Mortgage |

The sample data does not document currency or a data-collection source. Treat numeric monetary fields as demonstration values, not validated financial records.

## Models

| Model | Role |
|---|---|
| Support Vector Machine (SVM) | Model currently loaded by the Streamlit app |
| K-Nearest Neighbors (KNN) | Comparison model |
| Decision Tree | Comparison model |
| LightGBM | Comparison model |

Metrics vary with the small synthetic dataset. Previous local runs reported 1.00 test accuracy for SVM, KNN, and Decision Tree and 0.60 for LightGBM, using just five test rows. These are not comparable to a properly validated production model and should not be presented as evidence of high real-world accuracy.

## Explainability

The `src/xai_explanation.py` script calculates SHAP-based feature importance on the tiny held-out sample and writes outputs under `data/`. This is a demonstration of the XAI workflow only. With five test samples and synthetic data, feature rankings are unstable and do not establish causal effects.

## Workflow

```text
Applicant details
       |
       v
Saved preprocessing pipeline
(scaling + one-hot encoding)
       |
       v
Trained SVM model
       |
       v
Predicted class + model confidence
       |
       v
Demonstration decision label
```

The SHAP analysis runs as a separate script:

```text
Processed test data + trained SVM
       |
       v
SHAP explainer
       |
       v
Feature-importance CSV and plot
```

## Project Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── credit_risk.csv
│   ├── X_train_processed.csv
│   ├── X_test_processed.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── shap_feature_importance.csv
│   └── SHAP plot images
├── models/
│   ├── preprocessor.pkl
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── decision_tree_model.pkl
│   └── lightgbm_model.pkl
├── src/
│   ├── preprocessing.py
│   ├── train_svm.py
│   ├── train_knn.py
│   ├── train_decision_tree.py
│   ├── train_lightgbm.py
│   ├── model_comparison.py
│   ├── predict.py
│   └── xai_explanation.py
└── tests/
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/harshaharshu09/Explainable-Credit-Risk-Assessment-System.git
cd Explainable-Credit-Risk-Assessment-System
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).

### Run preprocessing, training, or SHAP separately

Execute these from the repository root:

```bash
python src/preprocessing.py
python src/train_svm.py
python src/train_knn.py
python src/train_decision_tree.py
python src/train_lightgbm.py
python src/model_comparison.py
python src/xai_explanation.py
```

Run preprocessing before the training scripts if regenerating artifacts. Training scripts may overwrite saved model files.

## Dependencies

Runtime and analysis dependencies are listed in `requirements.txt`: Streamlit, Pandas, NumPy, scikit-learn, Joblib, SHAP, Matplotlib, and LightGBM.

## Limitations and Responsible Use

- The included dataset is synthetic and contains only 25 records.
- The held-out test set has five rows; a single example changes accuracy by 20 percentage points.
- A predicted class and SVM `predict_proba` value are not a formal creditworthiness assessment or calibrated lending probability.
- The approval/rejection wording is a demonstration label, not a financial recommendation.
- The sample dataset and current results are not suitable for lending, underwriting, customer screening, or other consequential decisions.
- Before any realistic evaluation, use an appropriate, documented dataset; review data rights and privacy; check class balance and leakage; apply cross-validation and threshold analysis; test calibration and fairness; and have qualified stakeholders validate the system.

## Future Work

- Evaluate on a larger, documented credit-risk dataset
- Add stratified cross-validation and confidence intervals
- Compare precision, recall, F1, ROC-AUC, and PR-AUC, with special attention to default-class recall
- Add a reproducible training pipeline and automated tests
- Add per-applicant explanations to the Streamlit interface
- Publish a hosted demo and link it here after verification

## Author

**Harshavardhan Kumar**  
GitHub: [@harshaharshu09](https://github.com/harshaharshu09)

---

If you use or extend this project, please keep its synthetic-data and prototype limitations visible.
