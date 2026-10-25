import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "svm_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
IMPORTANCE_PATH = ROOT / "data" / "shap_feature_importance.csv"

st.set_page_config(page_title="Explainable Credit Risk Assessment", page_icon="💳", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
.hero {padding: 1.5rem 0 1rem 0;}
.hero h1 {font-size: 2.8rem; margin-bottom: .35rem;}
.hero p {font-size: 1.15rem; color: #aab1bd;}
.result-card {padding: 1.1rem 1.3rem; border-radius: 12px; margin: .4rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>Explainable Credit Risk Assessment System</h1><p>AI-powered credit risk prediction and loan decision system using machine learning and SHAP.</p></div>", unsafe_allow_html=True)

st.info("Educational prototype only. This project uses a small synthetic dataset and should not be used for real lending decisions.")

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

st.header("Applicant Details")

with st.form("credit_risk_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        income = st.number_input("Monthly Income", min_value=0.0, value=60000.0, step=5000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0, step=10000.0)
    with col2:
        loan_term = st.number_input("Loan Term (Months)", min_value=1, max_value=360, value=36, step=1)
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=720.0, step=10.0)
        existing_debt = st.number_input("Existing Debt", min_value=0.0, value=50000.0, step=5000.0)
    with col3:
        employment_years = st.number_input("Employment Years", min_value=0.0, max_value=60.0, value=8.0, step=1.0)
        loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Business", "Education", "Auto"])
        home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])

    submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True, type="primary")

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
        st.header("Credit Risk Result")
        result_col, confidence_col = st.columns([2, 1])

        with result_col:
            if prediction == 0:
                st.success("LOW RISK / NO DEFAULT")
                st.success("LOAN DECISION: APPROVED")
            else:
                st.error("DEFAULT RISK")
                st.error("LOAN DECISION: REJECTED")

        with confidence_col:
            if probabilities is not None:
                class_index = list(model.classes_).index(prediction)
                st.metric("Prediction Confidence", f"{probabilities[class_index] * 100:.2f}%")
                st.caption("Model confidence output; not a guarantee of repayment.")

        st.divider()
        st.header("Applicant Summary")
        summary = pd.DataFrame({
            "Field": ["Age", "Monthly Income", "Loan Amount", "Loan Term", "Credit Score", "Existing Debt", "Employment Years", "Loan Purpose", "Home Ownership"],
            "Value": [age, f"{income:,.0f}", f"{loan_amount:,.0f}", f"{loan_term} months", f"{credit_score:,.0f}", f"{existing_debt:,.0f}", f"{employment_years:g}", loan_purpose, home_ownership]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if IMPORTANCE_PATH.exists():
            st.divider()
            st.header("Model Explainability")
            importance = pd.read_csv(IMPORTANCE_PATH)
            st.caption("Global SHAP feature importance from the project's synthetic test data. These values are illustrative and are not applicant-specific explanations.")
            st.bar_chart(importance.head(10).set_index("Feature")["Importance"])

    except Exception as exc:
        st.error("Prediction failed. Check the saved preprocessing and model artifacts.")
        st.code(str(exc))

st.divider()
st.caption("Built with Streamlit • scikit-learn • LightGBM • Pandas • SHAP")
