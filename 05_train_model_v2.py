"""
Train V2 model with multiple features + SHAP explainability.

This is what separates "dashboard" from "decision intelligence".
SHAP shows judges exactly which factors drive the risk prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

# ---- Load engineered features ----
print("Loading engineered features...\n")
features_df = pd.read_csv("data/ky_features_engineered.csv", index_col=0)

# Separate features from label
X = features_df.drop("high_risk", axis=1)
y = features_df["high_risk"]

print(f"Features: {X.columns.tolist()}")
print(f"Counties: {len(X)}, High-risk: {y.sum()}\n")

# ---- Scale features (important for SHAP interpretation) ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# ---- Train model ----
print("Training logistic regression on 4 features...\n")
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, y_pred, target_names=["Stable", "High Risk"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---- SHAP explainability ----
print("\n\nComputing SHAP explanations...")
explainer = shap.LinearExplainer(model, X_train, feature_names=X.columns.tolist())
shap_values = explainer.shap_values(X_test)

print("✅ SHAP computed for all test samples\n")

# ---- Save everything ----
with open("model_v2.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Saved model to model_v2.pkl")

with open("scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Saved scaler to scaler_v2.pkl")

with open("explainer_v2.pkl", "wb") as f:
    pickle.dump(explainer, f)
print("✅ Saved SHAP explainer to explainer_v2.pkl")

# Save SHAP values for quick lookup
np.save("data/shap_values_v2.npy", shap_values)
print("✅ Saved SHAP values to data/shap_values_v2.npy")

# Save feature scaling reference
X_scaled.to_csv("data/X_scaled_reference.csv")
print("✅ Saved scaled feature reference to data/X_scaled_reference.csv")

# Save model coefficients for inspection
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0],
    "abs_importance": np.abs(model.coef_[0])
}).sort_values("abs_importance", ascending=False)

print("\n\nFeature Importance (Model Coefficients):")
print(coef_df)
coef_df.to_csv("data/model_coefficients.csv", index=False)

print("\n\n🎯 V2 Model complete! Next: Update app.py with SHAP explanations")
