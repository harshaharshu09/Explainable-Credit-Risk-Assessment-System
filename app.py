import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------
@st.cache_resource
def load_models():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/svm_model.pkl")
    return preprocessor, model

preprocessor, model = load_models()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Explainable Credit Risk Assessment System")
st.markdown(
    "### AI-powered credit risk prediction and loan decision system"
)

st.divider()

# --------------------------------------------------
# APPLICANT DETAILS
# --------------------------------------------------
st.subheader(" Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=60000.0,
        step=5000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=200000.0,
        step=10000.0
    )

with col2:
    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        max_value=360,
        value=36
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=720.0,
        step=10.0
    )

    existing_debt = st.number_input(
        "Existing Debt",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )

with col3:
    employment_years = st.number_input(
        "Employment Years",
        min_value=0.0,
        max_value=60.0,
        value=8.0,
        step=1.0
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Personal", "Home", "Business", "Education", "Auto"]
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        ["Rent", "Own", "Mortgage"]
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button(
    "Assess Credit Risk",
    use_container_width=True,
    type="primary"
):

    input_data = pd.DataFrame([{
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "employment_years": employment_years,
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership
    }])

    try:
        # Apply the same preprocessing used during training
        input_encoded = preprocessor.transform(input_data)

        # Keep exact feature names
        feature_names = preprocessor.get_feature_names_out()

        input_encoded_df = pd.DataFrame(
            input_encoded,
            columns=feature_names
        )

        # Prediction
        prediction = model.predict(input_encoded_df)[0]

        # Probability if available
        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(
                input_encoded_df
            )[0]

        st.divider()
        st.subheader(" Credit Risk Result")

        if prediction == 0:

            st.success(
                " LOW RISK / NO DEFAULT"
            )

            st.success(
                " LOAN DECISION: APPROVED"
            )

            if probability is not None:
                confidence = max(probability) * 100
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

        else:

            st.error(
                " HIGH RISK / DEFAULT"
            )

            st.error(
                " LOAN DECISION: REJECTED"
            )

            if probability is not None:
                confidence = max(probability) * 100
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

        # --------------------------------------------------
        # APPLICANT SUMMARY
        # --------------------------------------------------
        st.divider()
        st.subheader(" Applicant Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.write(f"**Age:** {age}")
            st.write(f"**Income:** {income:,.0f}")
            st.write(f"**Loan Amount:** {loan_amount:,.0f}")

        with summary_col2:
            st.write(f"**Credit Score:** {credit_score:.0f}")
            st.write(f"**Existing Debt:** {existing_debt:,.0f}")
            st.write(f"**Employment:** {employment_years:.0f} years")

        with summary_col3:
            st.write(f"**Loan Purpose:** {loan_purpose}")
            st.write(f"**Home Ownership:** {home_ownership}")
            st.write(f"**Loan Term:** {loan_term} months")

    except Exception as e:

        st.error("Prediction failed.")
        st.code(str(e))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Explainable Credit Risk Assessment System | "
    "Machine Learning + SHAP"
)

