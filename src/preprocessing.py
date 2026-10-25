import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("data/credit_risk.csv")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, "models/preprocessor.pkl")

pd.DataFrame(X_train_processed).to_csv(
    "data/X_train_processed.csv",
    index=False
)

pd.DataFrame(X_test_processed).to_csv(
    "data/X_test_processed.csv",
    index=False
)

pd.DataFrame(y_train).to_csv(
    "data/y_train.csv",
    index=False
)

pd.DataFrame(y_test).to_csv(
    "data/y_test.csv",
    index=False
)

print("Original data shape:", df.shape)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print("Preprocessor saved successfully!")
print("Processed files saved successfully!")
