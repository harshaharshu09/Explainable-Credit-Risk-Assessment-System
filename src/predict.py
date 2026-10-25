import pandas as pd
import joblib

print("=" * 60)
print("        CREDIT RISK PREDICTION SYSTEM")
print("=" * 60)

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/svm_model.pkl")

print("\nEnter applicant details:\n")

age = int(input("Age: "))
income = float(input("Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = int(input("Loan Term (months): "))
credit_score = float(input("Credit Score: "))
existing_debt = float(input("Existing Debt: "))
employment_years = float(input("Employment Years: "))

loan_purpose = input(
    "Loan Purpose (Personal/Home/Business): "
).strip()

home_ownership = input(
    "Home Ownership (Rent/Own/Mortgage): "
).strip()

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

input_encoded = preprocessor.transform(input_data)

# Keep feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()
input_encoded_df = pd.DataFrame(
    input_encoded,
    columns=feature_names
)

prediction = model.predict(input_encoded_df)[0]

print("\n" + "=" * 60)
print("              CREDIT RISK RESULT")
print("=" * 60)

if prediction == 0:
    print("Prediction: LOW RISK / NO DEFAULT")
    print("Loan Decision: APPROVED")
else:
    print("Prediction: HIGH RISK / DEFAULT")
    print("Loan Decision: REJECTED")

print("=" * 60)
print("Prediction completed successfully!")
