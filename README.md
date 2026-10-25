# Explainable Credit Risk Assessment System

A Streamlit-based machine-learning prototype for exploring loan-default risk and model explainability.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20LightGBM-orange)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-6236FF)](https://shap.readthedocs.io/)

> **Prototype / educational project.** This repository uses a manually created synthetic dataset containing 25 rows. The held-out test set has only five rows, so results are illustrative, not reliable estimates of real-world performance. Do not use this prototype for actual lending decisions.

## Overview

The application collects applicant and loan details, applies a saved preprocessing pipeline, and uses a trained Support Vector Machine (SVM) to predict the dataset label `default`. The repository also contains training scripts for a Decision Tree, K-Nearest Neighbors (KNN), LightGBM, and SVM, plus a separate SHAP feature-importance analysis.

## Features

- Streamlit interface for applicant information
- Numeric scaling and categorical one-hot encoding
- Binary prediction: `0 = no default`, `1 = default`
- Demonstration decision label derived from the predicted class
- SVM probability output presented as model confidence (not a calibrated repayment probability)
- Model training and comparison scripts
- Accuracy, precision, recall, F1-score, and confusion matrix reports
- SHAP-based global feature-importance output and plots

## Application

**Live demo:** Not published yet. Deployment instructions are provided below; add the hosted URL here after it is successfully published.

## Inputs

| Feature | Description |
|---|---|
| Age | Applicant age in years |
| Monthly income | Income value used by the sample dataset |
| Loan amount | Requested loan amount |
| Loan term | Duration in months |
| Credit score | Credit-score value |
| Existing debt | Existing debt value |
| Employment years | Years employed |
| Loan purpose | Personal, Home, Business, Education, or Auto |
| Home ownership | Rent, Own, or Mortgage |

The synthetic data does not document a currency or an external data source. Monetary values are demonstration values, not verified financial records.

## Models

| Model | Role |
|---|---|
| Support Vector Machine (SVM) | Model currently loaded by the Streamlit app |
| K-Nearest Neighbors (KNN) | Comparison model |
| Decision Tree | Comparison model |
| LightGBM | Comparison model |

Previous local runs reported 1.00 test accuracy for SVM, KNN, and Decision Tree and 0.60 for LightGBM on just five test rows. These scores are highly uncertain; a single test example changes accuracy by 20 percentage points. Do not present these results as evidence of real-world predictive performance.

## Explainable AI

The `src/xai_explanation.py` script uses SHAP to estimate global feature importance on the tiny held-out sample and saves output files in `data/`. With only five test examples and synthetic data, rankings can be unstable and should not be interpreted as causal effects.

## Workflow

```text
Applicant details
       |
       v
Saved preprocessing pipeline
(scaling + one-hot encoding)
       |
       v
Trained SVM
       |
       v
Predicted class + model confidence
       |
       v
Demonstration decision label
```

The global SHAP analysis runs separately:

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
├── runtime.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
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

### 1. Clone

```bash
git clone https://github.com/harshaharshu09/Explainable-Credit-Risk-Assessment-System.git
cd Explainable-Credit-Risk-Assessment-System
```

### 2. Create and activate an environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the app

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL, typically `http://localhost:8501`.

### Run analysis scripts

Run from the project root. Re-run preprocessing before retraining models if regenerating artifacts.

```bash
python src/preprocessing.py
python src/train_svm.py
python src/train_knn.py
python src/train_decision_tree.py
python src/train_lightgbm.py
python src/model_comparison.py
python src/xai_explanation.py
```

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in and connect your GitHub account, if asked.
3. Select **Create app** and choose repository `harshaharshu09/Explainable-Credit-Risk-Assessment-System`.
4. Select branch `main` and main file path `app.py`.
5. Select **Deploy** and wait for the build logs to report success.
6. Open the resulting public app URL and test an assessment. Add that verified URL to the **Live demo** line above.

The repository includes a dependency list and Python 3.11 runtime file. Hosting still requires the repository owner to authorize and launch the deployment. If the build fails, inspect deployment logs before changing dependencies.

## Dependencies

Dependencies are listed in `requirements.txt`: Streamlit, Pandas, NumPy, scikit-learn, Joblib, SHAP, Matplotlib, and LightGBM.

## Limitations and Responsible Use

- Dataset: 25 synthetic records, not real borrower data.
- Test set: five rows; reported metrics are too uncertain to support production claims.
- SVM `predict_proba` output is not necessarily calibrated and should not be described as the true probability that a borrower will repay.
- The app's “approved” or “rejected” label is a demonstration output, not a financial recommendation.
- The prototype is not suitable for lending, underwriting, customer screening, or other consequential decisions.
- Before realistic evaluation, use an appropriately licensed and documented dataset; inspect class balance and potential leakage; use cross-validation and threshold analysis; evaluate default-class recall, ROC-AUC and PR-AUC; assess calibration and fairness; and obtain qualified review.

## Future Work

- Evaluate on a larger, documented credit-risk dataset
- Add stratified cross-validation and uncertainty intervals
- Compare metrics with special attention to default-class recall
- Add automated tests and a reproducible training pipeline
- Add individual prediction explanations to the Streamlit interface
- Publish and verify a hosted demo

## Author

**Harshavardhan Kumar**  
GitHub: [@harshaharshu09](https://github.com/harshaharshu09)

---

Keep the synthetic-data and prototype limitations visible when presenting or extending this project.
