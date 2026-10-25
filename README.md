# Explainable Credit Risk Assessment System

AI-powered credit risk prediction and loan decision system using Machine Learning and SHAP.

## Project Overview

The **Explainable Credit Risk Assessment System** predicts whether a loan applicant is likely to be a low-risk or high-risk borrower and provides a simple loan decision through a Streamlit web application.

The system takes applicant information such as income, loan amount, loan term, credit score, existing debt, employment history, loan purpose, and home ownership. The input is processed using the same preprocessing pipeline used during model training and passed to a trained machine learning model.

## Key Features

- Credit risk prediction for loan applicants
- Low-risk / high-risk classification
- Automatic loan approval or rejection decision
- Prediction confidence when probability is available
- Applicant summary after assessment
- Streamlit-based interactive web interface
- Data preprocessing pipeline saved for reuse
- Multiple machine learning models included for comparison
- SHAP-based explainability components included in the project

## Input Features

The application uses the following applicant details:

| Feature | Description |
|---|---|
| Age | Applicant's age |
| Monthly Income | Applicant's monthly income |
| Loan Amount | Requested loan amount |
| Loan Term | Loan duration in months |
| Credit Score | Applicant's credit score |
| Existing Debt | Existing outstanding debt |
| Employment Years | Years of employment |
| Loan Purpose | Personal, Home, Business, Education, or Auto |
| Home Ownership | Rent, Own, or Mortgage |

## Machine Learning Models

The repository contains training and comparison components for:

- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- LightGBM

The current Streamlit application loads the trained **SVM model** and the saved preprocessing pipeline for prediction.

## Explainable AI

The project includes SHAP-related analysis files to support model explainability and feature-importance analysis. This helps understand which applicant characteristics contribute to credit-risk predictions.

## Project Structure

```text
Explainable-Credit-Risk-Assessment-System/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── credit_risk.csv
│   ├── X_train_processed.csv
│   ├── X_test_processed.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── shap_feature_importance.csv
│   ├── shap_feature_importance.png
│   └── shap_summary_plot.png
│
├── models/
│   ├── preprocessor.pkl
│   ├── svm_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   └── lightgbm_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   ├── model_comparison.py
│   ├── train_svm.py
│   ├── train_decision_tree.py
│   ├── train_knn.py
│   ├── train_lightgbm.py
│   └── xai_explanation.py
│
└── tests/
    └── __init__.py
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/harshaharshu09/Explainable-Credit-Risk-Assessment-System.git
cd Explainable-Credit-Risk-Assessment-System
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If dependencies have not yet been added to `requirements.txt`, install the main packages with:

```bash
pip install streamlit pandas scikit-learn joblib shap lightgbm
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Example Prediction

Example applicant:

- Age: 35
- Monthly Income: 60,000
- Loan Amount: 200,000
- Loan Term: 36 months
- Credit Score: 720
- Existing Debt: 50,000
- Employment: 8 years
- Loan Purpose: Personal
- Home Ownership: Own

The application can return a result such as:

```text
LOW RISK / NO DEFAULT
LOAN DECISION: APPROVED
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- SHAP
- LightGBM
- Machine Learning
- Explainable AI (XAI)

## Project Workflow

```text
Applicant Details
       ↓
Data Preprocessing
       ↓
Feature Encoding
       ↓
Trained ML Model
       ↓
Credit Risk Prediction
       ↓
Loan Decision
       ↓
Prediction Confidence
       ↓
Explainability / SHAP Analysis
```

## Future Improvements

- Add more advanced explainability visualizations directly to the web application
- Add model performance metrics and comparison charts
- Add probability-based risk scoring
- Add downloadable assessment reports
- Deploy the application to a cloud platform

## Author

**Harshavardhan Kumar**

GitHub: https://github.com/harshaharshu09

---

**Project:** Explainable Credit Risk Assessment System  
**Focus:** Machine Learning + Credit Risk Assessment + Explainable AI