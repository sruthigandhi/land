"""
Instead of vegetation health, use NDVI to show LAND COVER CHANGE.
This captures structural shifts: from active crop → idle/pasture → fallow.
"""

import pandas as pd
import numpy as np

features = pd.read_csv("data/ky_features_with_ndvi.csv", index_col=0)

# Land-cover-change proxy:
# If acreage is declining but farms aren't disappearing proportionally,
# land is shifting to lower-intensity use (pasture/fallow instead of cropland)

features["land_cover_shift_score"] = (
    (features["acreage_change_pct"] < features["farm_count_change_pct"]) * 
    100
).astype(int)

# Counties with >10% acreage loss = likely crop-to-pasture/fallow shift
features["crop_intensity_declining"] = (features["acreage_change_pct"] < -10).astype(int)

print("Land Cover Change Analysis:")
print(f"Counties shifting to lower-intensity use: {features['land_cover_shift_score'].sum()}")
print(f"Counties with significant crop decline: {features['crop_intensity_declining'].sum()}\n")

features.to_csv("data/ky_features_final.csv")
print("✅ Updated features with land-cover-change indicators")
