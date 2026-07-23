"""
V3: Random Forest model for better accuracy + feature importance.
RF captures non-linear relationships better than logistic regression.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load engineered features
print("Loading engineered features...\n")
features_df = pd.read_csv("data/ky_features_engineered.csv", index_col=0)

X = features_df.drop("high_risk", axis=1)
y = features_df["high_risk"]

print(f"Features: {X.columns.tolist()}")
print(f"Counties: {len(X)}, High-risk: {y.sum()}\n")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Train Random Forest with class balancing
print("Training Random Forest...\n")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, y_pred, target_names=["Stable", "High Risk"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n\nFeature Importance (Random Forest):")
print(importance_df)

# Save everything
with open("model_rf_v3.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n✅ Saved Random Forest model to model_rf_v3.pkl")

with open("scaler_rf_v3.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Saved scaler to scaler_rf_v3.pkl")

importance_df.to_csv("data/rf_feature_importance.csv", index=False)
print("✅ Saved feature importance to data/rf_feature_importance.csv")

print("\nNext: Add NDVI validation layer and update app")
