# Explainable Credit Risk Assessment System

An interactive **machine-learning based credit risk assessment application** that predicts default risk, provides a loan decision for demonstration purposes, and presents model confidence and global SHAP feature importance through a Streamlit interface.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-6236FF)](https://shap.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-9ACD32)](https://lightgbm.readthedocs.io/)

> **Educational prototype:** The project uses a small synthetic dataset of 25 records, with only five records in the held-out test set. Results are illustrative and are not reliable estimates of real-world lending performance. This application must not be used for actual lending, underwriting, or financial decisions.

## 🚀 Live Demo

**Try the application:** [Explainable Credit Risk Assessment System](https://harshaharshu09-explainable-credit-risk-assessment-sy-app-lbhfz0.streamlit.app/)

Enter applicant details in the web interface and click **Assess Credit Risk** to see the prediction, demonstration loan decision, confidence output, applicant summary, and global SHAP feature-importance chart.

## 📌 Project Overview

Credit risk assessment is the process of estimating whether a borrower is likely to default on a loan. This project demonstrates a complete machine-learning workflow for a simplified credit-risk use case:

1. Collect applicant and loan information.
2. Preprocess numerical and categorical features.
3. Transform the input using the saved preprocessing pipeline.
4. Apply a trained Support Vector Machine (SVM).
5. Predict the default-risk class.
6. Display a demonstration loan decision.
7. Show model confidence when probability output is available.
8. Present global SHAP feature importance for model interpretability.

## ✨ Key Features

- Interactive Streamlit web application
- Applicant-level credit-risk prediction
- Demonstration **Approved / Rejected** loan decision
- Saved preprocessing pipeline for consistent inference
- Numerical feature scaling
- Categorical one-hot encoding
- SVM-based prediction in the live application
- Model comparison scripts for SVM, KNN, Decision Tree, and LightGBM
- Accuracy, precision, recall, F1-score, and confusion-matrix evaluation
- Global SHAP feature-importance analysis
- Reproducible project structure with separate data, model, source, and test folders

## 🧠 Machine Learning Models

| Model | Purpose | Used by Live App |
|---|---|---|
| **Support Vector Machine (SVM)** | Primary classification model | ✅ Yes |
| **K-Nearest Neighbors (KNN)** | Model comparison | No |
| **Decision Tree** | Model comparison | No |
| **LightGBM** | Model comparison | No |

The Streamlit application loads the saved SVM model and preprocessing pipeline from the `models/` directory.

## 📊 Input Features

| Feature | Description |
|---|---|
| Age | Applicant age in years |
| Monthly Income | Monthly income used by the sample dataset |
| Loan Amount | Requested loan amount |
| Loan Term | Requested loan duration in months |
| Credit Score | Applicant credit-score value |
| Existing Debt | Existing debt value |
| Employment Years | Number of years employed |
| Loan Purpose | Personal, Home, Business, Education, or Auto |
| Home Ownership | Rent, Own, or Mortgage |

The project dataset is synthetic and does not represent real borrower records or a verified external financial dataset. Monetary values are demonstration values and no currency is assumed.

## 🔄 Application Workflow

```text
                 Applicant Details
                        │
                        ▼
             Saved Preprocessing Pipeline
             ┌──────────────────────────┐
             │ Numerical Scaling        │
             │ Categorical Encoding     │
             └──────────────────────────┘
                        │
                        ▼
                  Trained SVM
                        │
                        ▼
              Default-Risk Prediction
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          Low Risk            Default Risk
              │                   │
              ▼                   ▼
           Approved             Rejected
              │
              ▼
       Confidence + Summary
              │
              ▼
       Global SHAP Importance
```

## 🔍 Explainable AI with SHAP

The project includes a dedicated SHAP analysis script in `src/xai_explanation.py`.

SHAP is used to estimate the contribution of input features to model predictions and to produce a global feature-importance output. The generated artifacts are stored in the `data/` directory.

The current Streamlit interface displays the **global** SHAP feature importance chart. It is not an applicant-specific explanation for an individual prediction.

Because the underlying test set contains only five synthetic records, SHAP rankings should be treated as educational examples rather than stable real-world evidence.

## 🗂️ Project Structure

```text
Explainable-Credit-Risk-Assessment-System/
│
├── app.py                         # Streamlit application
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── data/
│   ├── credit_risk.csv            # Synthetic source dataset
│   ├── X_train_processed.csv      # Processed training features
│   ├── X_test_processed.csv       # Processed test features
│   ├── y_train.csv                # Training labels
│   ├── y_test.csv                 # Test labels
│   ├── shap_feature_importance.csv
│   └── SHAP output plots
│
├── models/
│   ├── preprocessor.pkl           # Saved preprocessing pipeline
│   ├── svm_model.pkl              # Saved SVM model
│   ├── knn_model.pkl              # Saved KNN model
│   ├── decision_tree_model.pkl    # Saved Decision Tree model
│   └── lightgbm_model.pkl         # Saved LightGBM model
│
├── src/
│   ├── preprocessing.py           # Data preprocessing
│   ├── train_svm.py               # SVM training
│   ├── train_knn.py               # KNN training
│   ├── train_decision_tree.py     # Decision Tree training
│   ├── train_lightgbm.py          # LightGBM training
│   ├── model_comparison.py        # Model evaluation/comparison
│   ├── predict.py                 # CLI prediction workflow
│   └── xai_explanation.py         # SHAP analysis
│
└── tests/
    └── __init__.py
```

## 🛠️ Tech Stack

- **Python** — application and machine-learning development
- **Streamlit** — interactive web interface
- **Pandas / NumPy** — data processing
- **Scikit-learn** — preprocessing, SVM, KNN, Decision Tree, and evaluation
- **LightGBM** — gradient-boosting model comparison
- **SHAP** — model explainability
- **Joblib** — model and preprocessing artifact persistence
- **Matplotlib** — visualization
- **Git & GitHub** — version control and project hosting

## ▶️ Run Locally

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

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```powershell
python -m streamlit run app.py
```

The terminal will provide a local address, normally:

```text
http://localhost:8501
```

## 🧪 Run the Machine Learning Pipeline

Run these commands from the project root when you want to regenerate the processed data, models, evaluation results, or SHAP artifacts.

```powershell
python src/preprocessing.py
python src/train_svm.py
python src/train_knn.py
python src/train_decision_tree.py
python src/train_lightgbm.py
python src/model_comparison.py
python src/xai_explanation.py
```

> If the saved model artifacts are regenerated, keep the training environment and deployment environment compatible with the installed dependency versions.

## 📈 Model Evaluation

The repository contains model-comparison code that evaluates the trained classifiers using common classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Earlier local experiments reported high scores for some models on the five-row test set. These values should **not** be interpreted as production-level performance because the evaluation sample is extremely small and synthetic. A single test example changes accuracy by 20 percentage points.

## ⚠️ Limitations & Responsible Use

This project is intentionally a learning and portfolio prototype.

- The dataset contains only 25 synthetic records.
- The held-out test set contains only five records.
- Small-sample metrics can be highly unstable.
- Synthetic data does not represent real borrower behavior.
- The SVM confidence output should not be interpreted as a guaranteed repayment probability.
- The application's approved/rejected label is a demonstration decision, not a financial recommendation.
- The system has not been validated for regulatory, fairness, bias, or real-world underwriting requirements.
- It should not be used for lending, credit approval, customer screening, or other consequential decisions.

For a production-grade credit-risk system, the workflow would require a properly sourced and licensed dataset, stronger validation, cross-validation, class-imbalance analysis, threshold tuning, calibration, fairness testing, monitoring, security controls, and qualified domain/regulatory review.

## 🚀 Future Enhancements

- Add a larger, documented real-world or appropriately licensed dataset
- Use stratified cross-validation and stronger statistical evaluation
- Add ROC-AUC and PR-AUC analysis
- Add probability calibration and threshold tuning
- Add applicant-specific SHAP explanations inside the Streamlit app
- Add automated unit and integration tests
- Add model versioning and reproducible training workflows
- Add input validation and stronger application error handling
- Add model monitoring and drift detection
- Improve the UI with risk-factor summaries and visual dashboards

## 👨‍💻 Author

**Harshavardhan Kumar**

- GitHub: [@harshaharshu09](https://github.com/harshaharshu09)
- Project: [Explainable Credit Risk Assessment System](https://github.com/harshaharshu09/Explainable-Credit-Risk-Assessment-System)
- Live Demo: [Streamlit App](https://harshaharshu09-explainable-credit-risk-assessment-sy-app-lbhfz0.streamlit.app/)

---

**Built for learning, experimentation, and demonstrating explainable machine learning in a credit-risk use case.**
