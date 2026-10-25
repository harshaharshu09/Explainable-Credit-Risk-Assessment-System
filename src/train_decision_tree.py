import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed data
X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Create Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("DECISION TREE CREDIT RISK MODEL")
print("================================")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/decision_tree_model.pkl")

print("\nDecision Tree model saved successfully!")
