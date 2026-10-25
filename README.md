# Explainable Credit Risk Assessment System

A Streamlit-based machine learning application for assessing loan-default risk and presenting model outputs in an explainable format using SHAP.

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn%20%7C%20LightGBM-orange)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-6236FF)](https://shap.readthedocs.io/)

> **Prototype / educational project.** The repository uses a manually created synthetic dataset containing 25 records. The held-out test set contains five records, so the reported metrics are illustrative rather than reliable estimates of real-world performance. This prototype must not be used for actual lending or underwriting decisions.

## Live Application

**Live Demo:** https://harshaharshu09-explainable-credit-risk-assessment-sy-app-lbhfz0.streamlit.app/

The live Streamlit application accepts applicant and loan information, applies the saved preprocessing pipeline, generates an SVM prediction, displays a demonstration loan decision and model confidence, and presents global SHAP feature-importance information.

## Overview

The project demonstrates an end-to-end, explainable credit-risk assessment workflow. The system takes applicant information as input, prepares the data using a reusable preprocessing pipeline, trains and compares multiple machine learning models, evaluates their performance, selects the SVM model for application inference, and uses SHAP to provide global model-level explainability.

## Features

- Interactive Streamlit interface for applicant and loan information
- Reusable preprocessing pipeline
- Numeric feature standardization with `StandardScaler`
- Categorical feature encoding with `OneHotEncoder`
- Binary default-risk prediction
- SVM probability output presented as model confidence
- Demonstration loan approval or rejection output
- Model comparison using SVM, KNN, Decision Tree, and LightGBM
- Accuracy, precision, recall, F1-score, and confusion matrix evaluation
- SHAP-based global feature-importance analysis
- Saved model and preprocessing artifacts for application inference
- Public Streamlit application

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

The complete system works through the following sequence.

### Step 1: Load the Dataset

The project starts with the synthetic credit-risk dataset stored in `data/credit_risk.csv`.

```text
credit_risk.csv
      |
      v
Load dataset with Pandas
      |
      v
Inspect applicant features and target
```

The target variable is `default`. The feature columns contain the applicant's financial, employment, loan, and housing information.

### Step 2: Separate Features and Target

The dataset is divided into:

```text
Dataset
  |
  +-----------------------------+
  |                             |
  v                             v
Features (X)                 Target (y)
Applicant information       default
                             0 / 1
```

`X` contains the input features used by the models, while `y` contains the default label.

### Step 3: Train-Test Split

The data is divided into training and test sets using an 80/20 stratified split.

```text
Complete Dataset
       |
       v
Stratified 80/20 Split
       |
       +----------------------+
       |                      |
       v                      v
Training Set              Test Set
80%                       20%
       |                      |
       |                      |
       v                      v
Used for training        Used for evaluation
```

Stratification keeps the class distribution consistent between the training and test portions as far as possible.

### Step 4: Identify Feature Types

The preprocessing pipeline separates the input columns into numeric and categorical features.

```text
Applicant Features
       |
       +-----------------------+
       |                       |
       v                       v
Numeric Features         Categorical Features
Age                      Loan Purpose
Income                   Home Ownership
Loan Amount
Loan Term
Credit Score
Existing Debt
Employment Years
```

### Step 5: Preprocess Numeric Features

Numeric columns are standardized using `StandardScaler`.

```text
Numeric Features
       |
       v
StandardScaler
       |
       v
Scaled Numeric Features
```

This puts numeric variables on a comparable scale before they are passed to the machine learning models.

### Step 6: Encode Categorical Features

Categorical columns are converted into machine-readable numerical features using `OneHotEncoder`.

```text
Categorical Features
       |
       v
OneHotEncoder
       |
       v
One-Hot Encoded Features
```

The fitted preprocessing pipeline is saved as `models/preprocessor.pkl` so that exactly the same transformation can be reused when a new applicant is entered in the application.

### Step 7: Generate the Processed Dataset

The numeric and categorical preprocessing branches are combined into one processed feature matrix.

```text
Scaled Numeric Features
          |
          +------------------+
                             |
One-Hot Encoded Features ----+
                             |
                             v
                    Processed Feature Matrix
                             |
                             v
                 Save preprocessing pipeline
                    preprocessor.pkl
```

Processed training and test data are also stored in the `data/` directory for model training and evaluation.

### Step 8: Train Multiple Machine Learning Models

The processed training data is used to train four classification models.

```text
Processed Training Data
          |
          +-----------+-----------+-----------+
          |           |           |           |
          v           v           v           v
        SVM         KNN     Decision Tree   LightGBM
          |           |           |           |
          +-----------+-----------+-----------+
                      |
                      v
              Compare Model Results
```

The project trains separate model artifacts for SVM, KNN, Decision Tree, and LightGBM.

### Step 9: Evaluate the Models

The trained models are evaluated using the held-out test data.

```text
Test Data
   |
   v
Trained Models
   |
   +-------------------------------+
   |                               |
   v                               v
Predictions                   Performance Metrics
                                      |
                       +--------------+--------------+
                       |              |              |
                       v              v              v
                    Accuracy      Precision       Recall
                       |
                       v
                       F1-score
                       |
                       v
                Confusion Matrix
```

The evaluation provides more information than accuracy alone and allows the model behavior to be inspected using multiple classification metrics.

### Step 10: Select the Application Model

The current Streamlit application uses the trained SVM model stored in `models/svm_model.pkl` for live inference.

```text
Trained SVM
    |
    v
models/svm_model.pkl
    |
    v
Loaded by Streamlit application
```

The other model artifacts remain available for comparison and experimentation.

### Step 11: Generate SHAP Explainability Results

SHAP is used to analyze global feature importance from the processed test data.

```text
Processed Test Data
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
        +---------------------------+
        |                           |
        v                           v
Feature Importance CSV        SHAP Summary Plot
        |                           |
        +-------------+-------------+
                      |
                      v
                   data/
```

The generated feature-importance results are stored in `data/shap_feature_importance.csv`, while the SHAP visualization is stored in the `data/` directory.

These are global explanations based on the project's synthetic test data. They are not applicant-specific explanations and should not be interpreted as causal effects.

### Step 12: Start the Streamlit Application

When `app.py` starts, it loads the saved preprocessing pipeline and SVM model.

```text
                    Streamlit Application
                            |
              +-------------+-------------+
              |                           |
              v                           v
     preprocessor.pkl                 svm_model.pkl
              |                           |
              v                           v
     Data transformation             Prediction
```

The application then displays the applicant input form.

### Step 13: Enter Applicant Details

The user provides the following information through the Streamlit interface:

```text
Age
Monthly Income
Loan Amount
Loan Term
Credit Score
Existing Debt
Employment Years
Loan Purpose
Home Ownership
```

The values are collected into a single-row Pandas DataFrame with the same feature structure expected by the preprocessing pipeline.

### Step 14: Apply the Saved Preprocessing Pipeline

The new applicant data is transformed using the already fitted `preprocessor.pkl`.

```text
New Applicant Data
        |
        v
Pandas DataFrame
        |
        v
Saved preprocessor.pkl
        |
        v
Same numeric scaling
        +
Same categorical encoding
        |
        v
Processed Applicant Features
```

This prevents the live application from using a different preprocessing method from the one used during model training.

### Step 15: Generate the SVM Prediction

The processed applicant features are passed to the saved SVM model.

```text
Processed Applicant Features
             |
             v
        SVM Model
             |
             +----------------------+
             |                      |
             v                      v
      Predicted Class        Probability Output
             |                      |
             |                      v
             |              Prediction Confidence
             |                      |
             +----------+-----------+
                        |
                        v
                 Decision Mapping
```

The model predicts one of two classes:

- `0` = No Default / Low Risk
- `1` = Default Risk

### Step 16: Convert Prediction into Demonstration Loan Decision

The current application maps the predicted class to a demonstration decision.

```text
Predicted Class
      |
      +---------------------------+
      |                           |
      v                           v
Class 0                      Class 1
Low Risk / No Default       Default Risk
      |                           |
      v                           v
APPROVED                    REJECTED
```

The approval or rejection displayed by the application is a demonstration output based on the model prediction. It is not a real lending decision.

### Step 17: Display Prediction Confidence

If probability output is available from the SVM model, the application displays the probability associated with the predicted class as **Prediction Confidence**.

This value is a model output and should not be interpreted as a guaranteed probability of repayment or as a calibrated lending probability.

### Step 18: Display Applicant Summary

After prediction, the application displays the submitted applicant information in a structured summary.

```text
Applicant Input
      |
      v
Applicant Summary
      |
      +-- Age
      +-- Monthly Income
      +-- Loan Amount
      +-- Loan Term
      +-- Credit Score
      +-- Existing Debt
      +-- Employment Years
      +-- Loan Purpose
      +-- Home Ownership
```

### Step 19: Display Global Model Explainability

When the SHAP feature-importance file is available, the application loads `data/shap_feature_importance.csv` and displays the leading global feature-importance values as a chart.

```text
Saved SHAP Feature Importance
             |
             v
      Load CSV in Streamlit
             |
             v
       Select top features
             |
             v
       Display bar chart
```

The chart explains which features were important to the model globally across the project's synthetic test data. It does not explain why one specific applicant received a particular prediction.

### Complete System Flow

```text
                    CREDIT RISK ASSESSMENT SYSTEM
                                  |
                                  v
                         Load credit_risk.csv
                                  |
                                  v
                      Separate X and target y
                                  |
                                  v
                       Stratified train/test split
                                  |
                                  v
                      Identify feature data types
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              Numeric Features            Categorical Features
                    |                           |
                    v                           v
             StandardScaler               OneHotEncoder
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                       Processed Feature Matrix
                                  |
                                  v
                         Train ML Classifiers
                                  |
             +--------------------+--------------------+
             |                    |                    |
             v                    v                    v
            SVM                  KNN            Decision Tree
             |                    |                    |
             +--------------------+--------------------+
                                  |
                                  v
                               LightGBM
                                  |
                                  v
                         Evaluate All Models
                                  |
                                  v
                       Select SVM for Application
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              Save SVM Model             Save Preprocessor
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                         Start Streamlit App
                                  |
                                  v
                      User enters applicant data
                                  |
                                  v
                     Apply saved preprocessing
                                  |
                                  v
                           SVM prediction
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              Predicted class             Probability
                    |                           |
                    v                           v
            Loan decision output        Model confidence
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                         Applicant summary
                                  |
                                  v
                      Global SHAP importance
                                  |
                                  v
                         Final result display
```

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
