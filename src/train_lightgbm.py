import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

df = pd.read_csv("data/credit_risk.csv")

X = df.drop("default", axis=1)
y = df["default"]

categorical_features = ["loan_purpose", "home_ownership"]

numerical_features = [
    "age",
    "income",
    "loan_amount",
    "loan_term",
    "credit_score",
    "existing_debt",
    "employment_years"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbosity=-1
)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)

print("================================")
print("LIGHTGBM CREDIT RISK MODEL")
print("================================")
print("Accuracy:", accuracy)
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/lightgbm_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print()
print("Model saved successfully!")
