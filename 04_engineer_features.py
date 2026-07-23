"""
Engineer multi-feature dataset from raw NASS pulls.

Features:
- Acreage change (%)
- Farm count change (%)
- Average farm size change (%)
- Operator concentration (proxy for consolidation)

This gives us the 'drivers' to explain with SHAP.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---- Load raw data ----
print("Loading cleaned NASS datasets...\n")

land = pd.read_csv("data/ky_land_in_farms_raw.csv")
farms = pd.read_csv("data/ky_farm_operations_raw.csv")

# Clean acreage data
land["Value"] = (
    land["Value"].astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
land = land[land["Value"].str.match(r"^\d+$", na=False)]
land["Value"] = land["Value"].astype(float)
land = land[["county_name", "year", "Value"]].rename(columns={"Value": "acres"})

# Clean farm operations data
farms["Value"] = (
    farms["Value"].astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
farms = farms[farms["Value"].str.match(r"^\d+$", na=False)]
farms["Value"] = farms["Value"].astype(float)
farms = farms[["county_name", "year", "Value"]].rename(columns={"Value": "num_farms"})

# ---- Pivot both to wide format ----
print("Computing feature changes (2017 → 2022)...\n")

land_pivot = land.pivot_table(
    index="county_name", columns="year", values="acres", aggfunc="mean"
)
land_pivot.columns = [f"acres_{int(col)}" for col in land_pivot.columns]

farms_pivot = farms.pivot_table(
    index="county_name", columns="year", values="num_farms", aggfunc="mean"
)
farms_pivot.columns = [f"farms_{int(col)}" for col in farms_pivot.columns]

# ---- Merge ----
features = pd.merge(land_pivot, farms_pivot, left_index=True, right_index=True)
features = features.dropna()

print(f"Counties with data: {len(features)}")
print(f"Columns: {features.columns.tolist()}\n")

# ---- Engineer features ----

# 1. Acreage change %
features["acreage_change_pct"] = (features["acres_2022"] - features["acres_2017"]) / features["acres_2017"] * 100

# 2. Farm count change %
features["farm_count_change_pct"] = (features["farms_2022"] - features["farms_2017"]) / features["farms_2017"] * 100

# 3. Average farm size (acres per farm)
features["avg_farm_size_2017"] = features["acres_2017"] / features["farms_2017"]
features["avg_farm_size_2022"] = features["acres_2022"] / features["farms_2022"]
features["avg_farm_size_change_pct"] = (features["avg_farm_size_2022"] - features["avg_farm_size_2017"]) / features["avg_farm_size_2017"] * 100

# 4. Consolidation index (shrinking farm count relative to land loss = consolidation)
features["consolidation_ratio"] = features["farm_count_change_pct"] / (features["acreage_change_pct"] + 0.1)

# Keep only engineered features
feature_cols = [
    "acreage_change_pct",
    "farm_count_change_pct", 
    "avg_farm_size_change_pct",
    "consolidation_ratio"
]
features = features[feature_cols]

print("Feature Summary:")
print(features.describe().round(2))

# ---- Create risk label (same as before) ----
print("\n\nCreating risk labels...\n")
threshold = features["acreage_change_pct"].quantile(0.25)
features["high_risk"] = (features["acreage_change_pct"] <= threshold).astype(int)

print(f"Risk threshold (25th percentile): {threshold:.1f}%")
print(f"High-risk counties: {features['high_risk'].sum()}\n")

# ---- Save ----
features.to_csv("data/ky_features_engineered.csv")
print("✅ Saved engineered features to data/ky_features_engineered.csv")
print("\nNext: 05_train_model_v2.py")
