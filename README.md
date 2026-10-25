# Explainable Credit Risk Assessment System

A Streamlit-based machine learning application for assessing loan-default risk and presenting model outputs in an explainable format using SHAP.

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn%20%7C%20LightGBM-orange)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-6236FF)](https://shap.readthedocs.io/)

> **Prototype / educational project.** The repository uses a manually created synthetic dataset containing 25 records. The held-out test set contains five records, so the reported metrics are illustrative rather than reliable estimates of real-world performance. This prototype must not be used for actual lending or underwriting decisions.

## Live Application

**Live Demo:** https://harshaharshu09-explainable-credit-risk-assessment-sy-app-lbhfz0.streamlit.app/

The deployed Streamlit application allows a user to enter applicant information and receive a model prediction, demonstration loan decision, prediction confidence, applicant summary, and global SHAP feature-importance visualization.

## Overview

The system is designed as an end-to-end credit-risk machine learning workflow:

1. Load the synthetic credit-risk dataset.
2. Separate applicant features from the target variable (`default`).
3. Split the data into training and test sets using a stratified 80/20 split.
4. Identify numeric and categorical features.
5. Standardize numeric features using `StandardScaler`.
6. Convert categorical features into one-hot encoded features using `OneHotEncoder`.
7. Save the fitted preprocessing pipeline for reuse during prediction.
8. Train multiple machine learning models for comparison.
9. Evaluate the models using classification metrics and a confusion matrix.
10. Save the selected SVM model as a reusable model artifact.
11. Run SHAP-based global feature-importance analysis on the processed test data.
12. Save the SHAP feature-importance results and visualization.
13. Load the saved preprocessor and SVM model in the Streamlit application.
14. Accept new applicant details through the web interface.
15. Apply the same preprocessing pipeline used during training.
16. Generate the SVM prediction and probability output.
17. Convert the predicted class into a demonstration loan decision.
18. Display the result, confidence, applicant summary, and global model-importance information.

## Features

- Interactive Streamlit interface for applicant and loan information
- Reusable preprocessing pipeline
- Numeric feature standardization
- Categorical feature one-hot encoding
- Binary default-risk prediction
- SVM probability output presented as model confidence
- Demonstration loan approval or rejection output
- Model comparison using SVM, KNN, Decision Tree, and LightGBM
- Accuracy, precision, recall, F1-score, and confusion matrix evaluation
- SHAP-based global feature-importance analysis
- Saved model and preprocessing artifacts for application inference
- Public Streamlit deployment

## Input Features

| Feature | Description |
|---|---|
| Age | Applicant age in years |
| Monthly Income | Monthly income value used by the sample dataset |
| Loan Amount | Requested loan amount |
| Loan Term | Requested repayment duration in months |
| Credit Score | Applicant credit-score value |
| Existing Debt | Existing debt value |
| Employment Years | Number of years employed |
| Loan Purpose | Personal, Home, Business, Education, or Auto |
| Home Ownership | Rent, Own, or Mortgage |

The synthetic dataset does not document a currency or an external data source. Monetary values are demonstration values and are not verified financial records.

## End-to-End Workflow

### Phase 1: Data Preparation

The source data is stored in `data/credit_risk.csv`.

The preprocessing script performs the following operations:

```text
credit_risk.csv
      |
      v
Separate features (X) and target (y)
      |
      v
Stratified 80/20 train-test split
      |
      +-------------------+
      |                   |
      v                   v
Training data         Test data
      |                   |
      +---------+---------+
                |
                v
Identify numeric and categorical columns
                |
        +-------+-------+
        |               |
        v               v
Numeric features   Categorical features
        |               |
        v               v
StandardScaler     OneHotEncoder
        |               |
        +-------+-------+
                |
                v
        Processed feature matrix
                |
                v
       Save preprocessor.pkl
```

The fitted preprocessing object is saved as `models/preprocessor.pkl`. This ensures that new applicant data is transformed using the same rules learned from the training data.

### Phase 2: Model Training

The processed training and test datasets are used to train and compare four models:

```text
Processed training data
          |
          +-------------------+-------------------+-------------------+
          |                   |                   |                   |
          v                   v                   v                   v
        SVM                 KNN            Decision Tree          LightGBM
          |                   |                   |                   |
          +-------------------+-------------------+-------------------+
                              |
                              v
                    Evaluate model performance
                              |
                              v
             Accuracy / Precision / Recall / F1
                              |
                              v
                       Confusion Matrix
```

The current Streamlit application loads the saved SVM model from `models/svm_model.pkl` for prediction. The other trained models are retained for comparison and experimentation.

### Phase 3: Model Evaluation

Each model is evaluated against the held-out test set. The project records classification metrics and a confusion matrix so that model behavior can be inspected beyond accuracy alone.

Because the test set contains only five records, each individual prediction has a large effect on the reported metrics. Therefore, these results should be treated only as a demonstration of the workflow.

### Phase 4: Explainable AI with SHAP

The XAI workflow is executed separately from the live prediction interface:

```text
Processed test data
        |
        v
Load trained SVM model
        |
        v
Create SHAP explainer
        |
        v
Calculate SHAP values
        |
        v
Calculate mean absolute feature importance
        |
        +-------------------------+
        |                         |
        v                         v
Feature importance CSV       SHAP summary plot
        |                         |
        +------------+------------+
                     |
                     v
          Stored in data/
```

The resulting global feature-importance file is saved as `data/shap_feature_importance.csv`, and the SHAP summary visualization is saved as `data/shap_summary_plot.png`.

These are global explanations based on the project's synthetic test data. They are not applicant-specific explanations and should not be interpreted as causal effects.

### Phase 5: Streamlit Application

When the deployed application starts, it loads two saved artifacts:

```text
models/preprocessor.pkl
          |
          v
   Preprocessing pipeline

models/svm_model.pkl
          |
          v
      Trained SVM
```

The user then enters the applicant details through the Streamlit form.

### Phase 6: Real-Time Prediction

For each submitted applicant, the application follows this sequence:

```text
User enters applicant details
            |
            v
Create one-row applicant DataFrame
            |
            v
Apply saved preprocessing pipeline
            |
            v
Generate transformed feature matrix
            |
            v
Pass features to trained SVM
            |
            +-----------------------+
            |                       |
            v                       v
     Predicted class          Probability output
            |                       |
            |                       v
            |              Prediction confidence
            |                       |
            +-----------+-----------+
                        |
                        v
              Demonstration decision
                        |
              +---------+---------+
              |                   |
              v                   v
       LOW RISK / NO DEFAULT    DEFAULT RISK
              |                   |
              v                   v
            APPROVED             REJECTED
```

The application then displays the applicant summary and, when available, the global SHAP feature-importance visualization.

## Prediction Logic

The model predicts the dataset target:

- `0` = No Default / Low Risk
- `1` = Default Risk

For the current application, the predicted class is converted into a demonstration decision:

- Predicted class `0` → `LOW RISK / NO DEFAULT` → `LOAN DECISION: APPROVED`
- Predicted class `1` → `DEFAULT RISK` → `LOAN DECISION: REJECTED`

The SVM probability output is displayed as **Prediction Confidence**. It is a model output and should not be interpreted as a guaranteed probability of repayment or a calibrated lending probability.

## Models

| Model | Role |
|---|---|
| Support Vector Machine (SVM) | Model currently used by the Streamlit application |
| K-Nearest Neighbors (KNN) | Comparison model |
| Decision Tree | Comparison model |
| LightGBM | Comparison model |

Previous local runs reported 1.00 test accuracy for SVM, KNN, and Decision Tree and 0.60 for LightGBM on five test rows. These values are highly uncertain because the test set is extremely small and synthetic.

## Project Structure

```text
Explainable-Credit-Risk-Assessment-System/
├── app.py
├── README.md
├── requirements.txt
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

## Local Setup

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

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Streamlit normally opens the application at `http://localhost:8501`.

### 5. Rebuild the machine learning artifacts when required

Run the scripts from the project root:

```bash
python src/preprocessing.py
python src/train_svm.py
python src/train_knn.py
python src/train_decision_tree.py
python src/train_lightgbm.py
python src/model_comparison.py
python src/xai_explanation.py
```

Run preprocessing before retraining models if the processed datasets or preprocessing configuration need to be regenerated.

## Deployment

The application is deployed using Streamlit Community Cloud from the `main` branch with `app.py` as the application entry point.

**Live Application:** https://harshaharshu09-explainable-credit-risk-assessment-sy-app-lbhfz0.streamlit.app/

For a new deployment, select the repository `harshaharshu09/Explainable-Credit-Risk-Assessment-System`, branch `main`, and file path `app.py`. Installable Python dependencies are provided in `requirements.txt`.

## Dependencies

The project uses the following main libraries:

- Streamlit
- Pandas
- NumPy
- scikit-learn
- Joblib
- SHAP
- Matplotlib
- LightGBM

Exact dependency ranges are defined in `requirements.txt`.

## Limitations and Responsible Use

- The dataset contains only 25 synthetic records.
- The held-out test set contains only five records.
- Reported evaluation metrics are too uncertain to support production claims.
- SVM `predict_proba` output is not necessarily calibrated and must not be described as the true probability that a borrower will repay.
- The application's approved or rejected label is a demonstration output, not a financial recommendation.
- The prototype is not suitable for lending, underwriting, customer screening, or other consequential decisions.
- A production system would require a larger, properly documented and appropriately licensed dataset, robust validation, threshold analysis, calibration, fairness assessment, security controls, monitoring, and qualified domain review.

## Future Improvements

- Replace the synthetic dataset with a larger, documented credit-risk dataset.
- Add stratified cross-validation and uncertainty analysis.
- Evaluate default-class recall, ROC-AUC, PR-AUC, and calibration.
- Add automated tests and reproducible training workflows.
- Add applicant-level SHAP explanations to the Streamlit interface.
- Add model versioning, monitoring, and data-drift checks.
- Improve the decision layer so approval thresholds are explicitly configurable and independently validated.

## Author

**Harshavardhan Kumar**  
GitHub: https://github.com/harshaharshu09
