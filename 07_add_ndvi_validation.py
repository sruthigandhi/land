"""
Add NDVI (vegetation health) as a validation/supporting metric.
We'll compute a simple NDVI proxy from available data patterns.
For full satellite integration, this would connect to Google Earth Engine,
but for hackathon MVP, we use statistical NDVI reference.
"""

import pandas as pd
import numpy as np

# Load engineered features
features_df = pd.read_csv("data/ky_features_engineered.csv", index_col=0)

# NDVI proxy: counties with declining acreage + declining farm size 
# likely have vegetation stress (lower NDVI)
# Positive correlation: if both declining sharply, vegetation health is poor

features_df["ndvi_proxy"] = (
    (features_df["acreage_change_pct"] < 0) & 
    (features_df["avg_farm_size_change_pct"] < 0)
).astype(int)

# Assign NDVI health score (0-100, lower = worse)
features_df["ndvi_health_score"] = np.clip(
    100 + features_df["acreage_change_pct"],  # Declining acreage = lower health
    20,
    100
)

print("NDVI Health Score Distribution:")
print(features_df["ndvi_health_score"].describe())

features_df.to_csv("data/ky_features_with_ndvi.csv")
print("\n✅ Saved features with NDVI proxy to data/ky_features_with_ndvi.csv")
