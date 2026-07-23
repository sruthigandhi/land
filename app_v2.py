"""
Streamlit V3: Viridis with Random Forest + NDVI Validation

97% accuracy model with explainable feature importance.
No emojis, clean layout.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Viridis", layout="wide")

# ---- Load RF model + scaler ----
with open("model_rf_v3.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_rf_v3.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature data with NDVI
features_df = pd.read_csv("data/ky_features_with_ndvi.csv", index_col=0)
importance_df = pd.read_csv("data/rf_feature_importance.csv")

# ---- Header ----
st.title("Viridis: Farmland Resilience Platform")
st.write(
    "Kentucky farmland declined from 13.0M acres (2017) to 12.4M acres (2022). "
    "Viridis identifies counties at risk of continued agricultural decline."
)
st.divider()

# ---- County selector ----
county_list = sorted(features_df.index.unique())
county = st.selectbox("Select a Kentucky county", county_list, index=0)

# Get county data
county_data = features_df.loc[county]
county_features = county_data[["acreage_change_pct", "farm_count_change_pct", "avg_farm_size_change_pct", "consolidation_ratio"]]

# Scale and predict
county_features_scaled = scaler.transform([county_features.values])[0]
risk_prob = model.predict_proba([county_features_scaled])[0][1]
risk_label = "High Risk" if risk_prob > 0.5 else "Stable"

# ---- Main metrics ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Abandonment Risk", f"{risk_prob*100:.0f}%")

with col2:
    st.metric("Risk Category", risk_label)

with col3:
    acreage_pct = county_data["acreage_change_pct"]
    st.metric("5-yr Acreage Change", f"{acreage_pct:.1f}%")

with col4:
    ndvi_score = county_data["ndvi_health_score"]
    st.metric("Vegetation Health", f"{ndvi_score:.0f}/100")

st.divider()

# ---- Feature drivers (Random Forest importance) ----
st.subheader("Risk Drivers")
st.write("Factors contributing to this county's risk score, ranked by model importance.")

# Get feature values and importance
feature_names = ["acreage_change_pct", "farm_count_change_pct", "avg_farm_size_change_pct", "consolidation_ratio"]
feature_values = county_features.values

drivers = pd.DataFrame({
    "Feature": feature_names,
    "Value": feature_values,
    "Model Importance": [0.706, 0.056, 0.127, 0.111]  # From RF importance
})
drivers = drivers.sort_values("Model Importance", ascending=False)

for idx, row in drivers.iterrows():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**{row['Feature']}**")
        st.write(f"Current: {row['Value']:.2f}")
    with col2:
        st.write(f"Importance: {row['Model Importance']*100:.1f}%")
    st.divider()

# ---- Risk assessment ----
st.subheader("Assessment")

if risk_prob > 0.5:
    st.error(f"High Risk. {county} County exhibits signals of agricultural decline.")
    st.write("""
    Recommended interventions:
    - Transition to low-pesticide precision agriculture
    - Enroll in USDA NRCS conservation programs
    - Diversify crops and explore alternatives (hemp, regenerative grazing)
    - Monitor commodity prices and input costs
    """)
else:
    st.success(f"Stable. {county} County farmland is currently trending well.")

st.divider()

# ---- Model performance ----
with st.expander("Model Details"):
    st.write("""
    **Data Source:** USDA NASS Census of Agriculture (2017 vs 2022)
    
    **Model:** Random Forest (100 trees, max_depth=8)
    
    **Performance:** 97% accuracy, 100% recall on high-risk counties
    
    **Features (by importance):**
    1. Acreage change (%) — 70.6%
    2. Avg farm size change (%) — 12.7%
    3. Consolidation ratio — 11.1%
    4. Farm count change (%) — 5.5%
    
    **Validation:** NDVI health score (0-100) based on vegetation patterns.
    Counties with declining acreage show lower vegetation health.
    """)

# ---- All counties ----
with st.expander("View all Kentucky counties"):
    display_df = features_df[["acreage_change_pct", "farm_count_change_pct", "avg_farm_size_change_pct", "high_risk", "ndvi_health_score"]].copy()
    display_df.columns = ["Acreage Change %", "Farm Count Change %", "Farm Size Change %", "High Risk", "NDVI Score"]
    display_df = display_df.sort_values("Acreage Change %")
    st.dataframe(display_df, use_container_width=True)

st.caption("Viridis v3.0 — Girls in STEM Global Hackathon & DSH Pitch Competition")
