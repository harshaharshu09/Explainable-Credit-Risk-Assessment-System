import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("              TRAIN SVM MODEL")
print("=" * 60)

# Load processed data
X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Load original preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Get the exact feature names created by preprocessing
feature_names = preprocessor.get_feature_names_out()

# Assign correct feature names to processed data
X_train.columns = feature_names
X_test.columns = feature_names

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Train SVM
model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nSVM CREDIT RISK MODEL")
print("=" * 40)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/svm_model.pkl")

print("\nSVM model saved successfully!")
print("=" * 60)
