# Explainable Credit Risk Assessment System

An end-to-end machine learning project that demonstrates credit-default risk classification, model comparison, and model explainability through an interactive Streamlit application.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-6236FF" alt="SHAP" />
  <img src="https://img.shields.io/badge/Status-Educational%20Prototype-lightgrey" alt="Educational prototype" />
</p>

> **Important:** This is an educational prototype built with a manually created synthetic dataset of just 25 records. Its five-row test set is far too small to support reliable performance claims. Do not use this application for real loan approvals, underwriting, or other consequential financial decisions.

## Live Demo

**[Open the Streamlit App](https://harshaharshu09-explainable-credit-risk-assessment-sy-app-lbhfz0.streamlit.app/)**

The app accepts sample applicant and loan details, applies the saved preprocessing pipeline, predicts a default-risk class with a trained SVM model, and displays a demonstration decision and global SHAP feature-importance chart.

## Project Overview

Credit-risk assessment is a classification problem in which applicant information is used to estimate whether a record belongs to a default or non-default class. This project demonstrates a complete machine-learning workflow: preparing data, training and comparing classifiers, evaluating model outputs, saving reusable artifacts, serving predictions through Streamlit, and examining global feature importance with SHAP.

The current web application uses **Support Vector Machine (SVM)** for inference. K-Nearest Neighbors (KNN), Decision Tree, and LightGBM are included for model comparison.

## Key Features

- Interactive Streamlit interface for entering applicant and loan information
- Numeric feature scaling with `StandardScaler`
- Categorical feature encoding with `OneHotEncoder`
- Reusable preprocessing and saved model artifacts
- Binary default-risk prediction using SVM
- Demonstration approval/rejection mapping for predicted classes
- Model comparison across SVM, KNN, Decision Tree, and LightGBM
- Evaluation scripts for accuracy, precision, recall, F1-score, and confusion matrix
- Global SHAP feature-importance analysis and visualizations
- Local setup and reproducible script-based workflow

## Technology Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn, LightGBM |
| Preprocessing | StandardScaler, OneHotEncoder |
| Explainable AI | SHAP |
| Visualization | Matplotlib, Streamlit charts |
| Web interface | Streamlit |
| Model persistence | Joblib |

Dependency version ranges are documented in `requirements.txt`.

## Applicant Input Features

The demo form accepts the following fields:

| Feature | Description |
|---|---|
| Age | Applicant age in years |
| Monthly income | Income value in the sample dataset |
| Loan amount | Requested loan amount |
| Loan term | Requested repayment duration in months |
| Credit score | Applicant credit-score value |
| Existing debt | Existing debt value |
| Employment years | Years of employment |
| Loan purpose | Personal, Home, Business, Education, or Auto |
| Home ownership | Rent, Own, or Mortgage |

The repository's synthetic data does not specify a verified currency or a real-world data source; monetary values are for demonstration only.

## How It Works

```text
Synthetic credit-risk dataset
            |
            v
   Separate features and target
            |
            v
     Stratified train/test split
            |
            v
       Preprocessing pipeline
      /                     \\
     v                       v
Scale numeric data     Encode categories
      \\                     /
       v                   v
        Processed features
                |
                v
       Train classification models
                |
                v
       Compare evaluation metrics
                |
                v
       Save model and preprocessor
                |
                v
        Streamlit applicant form
                |
                v
     Transform new applicant input
                |
                v
           SVM prediction
                |
       +--------+---------+
       |                  |
       v                  v
 Predicted class    Model probability
       |                  |
       v                  v
 Demonstration      Displayed model
 loan decision      confidence output
       |
       v
 Global SHAP feature-importance chart
```

### Workflow Details

1. **Load and inspect data:** Read `data/credit_risk.csv` with Pandas. The target column is `default`.
2. **Split features and target:** Separate applicant attributes (`X`) from the binary target (`y`).
3. **Create train/test sets:** Use the project's stratified 80/20 split for training and evaluation.
4. **Preprocess inputs:** Standardize numeric columns and one-hot encode categorical columns using the reusable preprocessing pipeline.
5. **Train and compare models:** Train SVM, KNN, Decision Tree, and LightGBM classifiers, then compare their evaluation metrics.
6. **Run application inference:** Load `models/svm_model.pkl` and `models/preprocessor.pkl` in the Streamlit app. Apply the fitted preprocessor to new applicant inputs before prediction.
7. **Explain model behavior:** Load the saved SHAP feature-importance results and display a global chart for the synthetic test data.

## Models

| Model | Purpose |
|---|---|
| Support Vector Machine (SVM) | Used by the current Streamlit application |
| K-Nearest Neighbors (KNN) | Comparison model |
| Decision Tree | Comparison model |
| LightGBM | Comparison model |

Evaluation scripts calculate metrics including accuracy, precision, recall, F1-score, and the confusion matrix. Because the current held-out test set has only five records, individual predictions can change metrics substantially. Do not interpret the current results as evidence of real-world effectiveness.

## Understanding the Prediction

The target labels are interpreted in the project as:

- `0`: No Default / Low Risk
- `1`: Default Risk

For demonstration, the application maps class `0` to **APPROVED** and class `1` to **REJECTED**. This is a simple prototype rule, not a validated lending policy.

If available, the SVM's probability output is shown as **Prediction Confidence**. It is a model output and is not necessarily calibrated; it must not be treated as a guaranteed repayment probability.

## Explainability with SHAP

The explainability script calculates global feature importance and saves its results in `data/shap_feature_importance.csv`. Visualization files are stored in `data/`.

The current dashboard displays **global** importance based on synthetic test data. It does not provide an applicant-specific explanation, prove causation, or establish that a feature should determine a lending outcome.

## Project Structure

```text
Explainable-Credit-Risk-Assessment-System/
├── app.py
├── requirements.txt
├── README.md
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
│   ├── shap_feature_importance.png
│   └── shap_summary_plot.png
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

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

The application URL will be displayed by Streamlit in the terminal.

## Rebuild the Model Artifacts

Run commands from the repository root. If the processed datasets or preprocessing configuration must be regenerated, run preprocessing before training:

```bash
python src/preprocessing.py
python src/train_svm.py
python src/train_knn.py
python src/train_decision_tree.py
python src/train_lightgbm.py
python src/model_comparison.py
python src/xai_explanation.py
```

The Streamlit app uses the saved SVM model and preprocessor. Retraining should be followed by checking that the saved artifacts are compatible and that the app runs successfully.

## Responsible Use and Limitations

- **Small synthetic dataset:** 25 rows with only five held-out test examples.
- **Unreliable performance estimates:** test metrics are illustrative and not statistically robust.
- **Not for real decisions:** do not use this prototype for lending, underwriting, customer screening, or other consequential financial decisions.
- **Probability interpretation:** model confidence is not a guarantee or necessarily a calibrated probability.
- **Limited explainability:** SHAP output is global and based on synthetic test data; it is not applicant-specific and does not imply causation.
- **Production requirements:** a real system would need a larger, documented and appropriately licensed dataset; robust validation; calibration; decision-threshold analysis; fairness and bias assessment; security and privacy controls; monitoring; and expert review.

## Potential Improvements

- Use a larger, documented, appropriately licensed dataset
- Add stratified cross-validation and more robust performance estimates
- Evaluate ROC-AUC, PR-AUC, class-specific recall, and probability calibration
- Add tests for preprocessing, inference, and the application interface
- Develop applicant-level explanations with appropriate validation
- Add model versioning, monitoring, and data-drift checks
- Separate model predictions from configurable, independently validated decision rules

## Author

**Harshavardhan Kumar**  
[GitHub Profile](https://github.com/harshaharshu09)

---

*Built as a machine-learning and explainable-AI learning project.*
