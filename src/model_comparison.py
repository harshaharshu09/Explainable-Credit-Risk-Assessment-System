import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 50)
print("CREDIT RISK MODEL COMPARISON")
print("=" * 50)

# Load processed test data
X_test = pd.read_csv("data/X_test_processed.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

print("\nTest data loaded successfully")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

models = {
    "SVM": "models/svm_model.pkl",
    "KNN": "models/knn_model.pkl",
    "Decision Tree": "models/decision_tree_model.pkl",
    "LightGBM": "models/lightgbm_model.pkl"
}

results = {}

for name, path in models.items():

    print("\n" + "=" * 50)
    print(f"Evaluating {name}...")
    print("=" * 50)

    try:
        model = joblib.load(path)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"Accuracy: {accuracy:.2f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    except Exception as e:
        print(f"Error while evaluating {name}: {e}")

print("\n" + "=" * 50)
print("MODEL COMPARISON RESULTS")
print("=" * 50)

for model_name, accuracy in results.items():
    print(f"{model_name:<20} {accuracy:.2f}")

if results:
    best_model = max(results, key=results.get)
    best_accuracy = results[best_model]

    print("\n" + "=" * 50)
    print("BEST MODEL")
    print("=" * 50)
    print(f"Model: {best_model}")
    print(f"Accuracy: {best_accuracy:.2f}")

    print("\nModel comparison completed successfully!")
else:
    print("\nNo models were successfully evaluated.")
