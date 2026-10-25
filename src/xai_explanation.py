import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

print("=" * 60)
print("        EXPLAINABLE CREDIT RISK ASSESSMENT")
print("=" * 60)

# Load processed test data
X_test = pd.read_csv("data/x_test_processed.csv")

print("\nTest data loaded successfully")
print("Test data shape:", X_test.shape)

# Load trained SVM model and preprocessor
model = joblib.load("models/svm_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

print("\nCreating SHAP explainer...")

# Get actual transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Clean ColumnTransformer prefixes
feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in feature_names
]

# SHAP permutation explainer
explainer = shap.Explainer(model.predict, X_test)

shap_values = explainer(X_test)

print("\nSHAP analysis completed successfully!")

# Convert SHAP values
values = np.asarray(shap_values.values)

if values.ndim == 3:
    values = values[:, :, 0]

# Mean absolute SHAP importance
importance = np.mean(np.abs(values), axis=0)
importance = np.asarray(importance).reshape(-1)

# Safety check
if len(feature_names) != len(importance):
    feature_names = [f"Feature_{i}" for i in range(len(importance))]

# Feature importance dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print("=" * 60)
print(importance_df.to_string(index=False))

# Save importance
importance_df.to_csv(
    "data/shap_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")

# SHAP graph
plt.figure()

shap.summary_plot(
    values,
    X_test,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "data/shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP graph saved successfully!")

print("\n" + "=" * 60)
print("XAI EXPLANATION COMPLETED SUCCESSFULLY!")
print("=" * 60)
