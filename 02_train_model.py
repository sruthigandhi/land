"""
Cleans NASS data, engineers features, creates a proxy abandonment risk label,
and trains a logistic regression model.

Run this after 01_pull_data.py has produced data/ky_land_in_farms_raw.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# ---- 1. Load and clean ----
print("Loading raw NASS data...")
land = pd.read_csv("data/ky_land_in_farms_raw.csv")

print(f"Raw data shape: {land.shape}")
print(f"Columns: {land.columns.tolist()}\n")

# NASS returns messy strings for Value (commas, "(D)" for withheld data) -- clean it
land["Value"] = (
    land["Value"].astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

# Drop rows where Value is not numeric (withheld, NA, etc)
land = land[land["Value"].str.match(r"^\d+$", na=False)]
land["Value"] = land["Value"].astype(float)

# Keep only what we need: county, year, acreage
land = land[["county_name", "year", "Value"]].rename(columns={"Value": "acres"})

print(f"Cleaned data shape: {land.shape}\n")

# ---- 2. Pivot to wide format and calculate acreage change ----
print("Computing acreage change from 2017 to 2022...\n")

pivot = land.pivot_table(index="county_name", columns="year", values="acres", aggfunc="mean")
pivot.columns = pivot.columns.astype(int)

# Calculate percent change
pivot["pct_change_5yr"] = (pivot[2022] - pivot[2017]) / pivot[2017] * 100
pivot = pivot.dropna(subset=["pct_change_5yr"])

print(f"Counties with valid data: {len(pivot)}\n")
print(f"Acreage change summary (2017 → 2022):")
print(f"  Mean: {pivot['pct_change_5yr'].mean():.1f}%")
print(f"  Median: {pivot['pct_change_5yr'].median():.1f}%")
print(f"  Min: {pivot['pct_change_5yr'].min():.1f}%")
print(f"  Max: {pivot['pct_change_5yr'].max():.1f}%\n")

# ---- 3. Create proxy label: worst quartile = "high risk" ----
threshold = pivot["pct_change_5yr"].quantile(0.25)  # bottom 25% = steepest decline
pivot["high_risk"] = (pivot["pct_change_5yr"] <= threshold).astype(int)

print(f"Risk label threshold (25th percentile): {threshold:.1f}%")
print(f"Counties labeled 'high risk': {pivot['high_risk'].sum()}\n")

# ---- 4. Train model ----
print("Training logistic regression model...\n")

X = pivot[["pct_change_5yr"]].fillna(0)
y = pivot["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# ---- 5. Evaluate ----
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("Model Performance on Test Set:")
print(classification_report(y_test, y_pred, target_names=["Stable", "High Risk"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---- 6. Save everything ----
pivot.to_csv("data/ky_county_risk_scores.csv")
print("\n✅ Saved county risk scores to data/ky_county_risk_scores.csv")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Saved trained model to model.pkl")

print("\n🎯 Next step: build the Streamlit app (app.py) to make this interactive!")
