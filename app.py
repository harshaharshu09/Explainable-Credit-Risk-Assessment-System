import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "svm_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
IMPORTANCE_PATH = ROOT / "data" / "shap_feature_importance.csv"

st.set_page_config(page_title="Explainable Credit Risk Assessment", layout="wide")

st.title("Explainable Credit Risk Assessment System")
st.markdown("Machine-learning prototype for exploring loan-default risk and model explainability.")
st.warning(
    "Educational prototype only: this app uses a synthetic dataset of 25 rows. "
    "It is not validated for real lending decisions."
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

st.subheader("Applicant details")
with st.form("credit_risk_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        income = st.number_input("Monthly income (sample data units)", min_value=0.0, value=60000.0, step=5000.0)
        loan_amount = st.number_input("Loan amount (sample data units)", min_value=0.0, value=200000.0, step=10000.0)
    with col2:
        loan_term = st.number_input("Loan term (months)", min_value=1, max_value=360, value=36, step=1)
        credit_score = st.number_input("Credit score", min_value=300.0, max_value=900.0, value=720.0, step=10.0)
        existing_debt = st.number_input("Existing debt (sample data units)", min_value=0.0, value=50000.0, step=5000.0)
    with col3:
        employment_years = st.number_input("Employment years", min_value=0.0, max_value=60.0, value=8.0, step=1.0)
        loan_purpose = st.selectbox("Loan purpose", ["Personal", "Home", "Business", "Education", "Auto"])
        home_ownership = st.selectbox("Home ownership", ["Rent", "Own", "Mortgage"])
    submitted = st.form_submit_button("Assess credit risk", use_container_width=True, type="primary")

if submitted:
    applicant = pd.DataFrame([{
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "employment_years": employment_years,
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
    }])

    try:
        encoded = preprocessor.transform(applicant)
        feature_names = preprocessor.get_feature_names_out()
        encoded_df = pd.DataFrame(encoded, columns=feature_names)
        prediction = int(model.predict(encoded_df)[0])
        probabilities = model.predict_proba(encoded_df)[0] if hasattr(model, "predict_proba") else None

        st.divider()
        st.subheader("Assessment result")
        result_col, confidence_col = st.columns(2)
        with result_col:
            if prediction == 0:
                st.success("Predicted class: LOW RISK / NO DEFAULT")
                st.info("Demonstration label: APPROVED")
            else:
                st.error("Predicted class: DEFAULT RISK")
                st.warning("Demonstration label: REJECTED")
        with confidence_col:
            if probabilities is not None:
                class_index = list(model.classes_).index(prediction)
                st.metric("Model output for predicted class", f"{probabilities[class_index] * 100:.1f}%")
                st.caption("SVM probability output is not calibrated and is not a probability of repayment.")

        st.subheader("Applicant summary")
        st.dataframe(applicant, use_container_width=True, hide_index=True)

        if IMPORTANCE_PATH.exists():
            st.subheader("Global SHAP feature importance")
            importance = pd.read_csv(IMPORTANCE_PATH)
            st.caption("These global values were computed separately on five synthetic test rows. They are illustrative, not applicant-specific explanations.")
            st.bar_chart(importance.head(10).set_index("Feature")["Importance"])

    except Exception as exc:
        st.error("Prediction failed. Check the saved preprocessing and model artifacts.")
        st.code(str(exc))

st.divider()
st.caption("Built with Streamlit, scikit-learn, LightGBM, Pandas, and SHAP.")
