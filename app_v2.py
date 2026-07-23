"""
Streamlit V2: Viridis with SHAP Explainability

Shows county risk + drivers (why it's at risk) using SHAP feature importance.
No emojis, clean layout.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

st.set_page_config(page_title="Viridis", layout="wide")

# ---- Load model, scaler, explainer ----
with open("model_v2.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_v2.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("explainer_v2.pkl", "rb") as f:
    explainer = pickle.load(f)

# Load feature data
features_df = pd.read_csv("data/ky_features_engineered.csv", index_col=0)
X_ref = pd.read_csv("data/X_scaled_reference.csv", index_col=0)

# ---- Header ----
st.title("Viridis: Farmland Resilience Platform")
st.write(
    "Kentucky farmland declined from 13.0M acres (2017) to 12.4M acres (2022). "
    "Viridis identifies counties at risk of continued agricultural decline and "
    "reveals the drivers behind each prediction."
)
st.divider()

# ---- County selector ----
county_list = sorted(features_df.index.unique())
county = st.selectbox("Select a Kentucky county", county_list, index=0)

# Get county data
county_data = features_df.loc[county]
county_features = county_data[["acreage_change_pct", "farm_count_change_pct", "avg_farm_size_change_pct", "consolidation_ratio"]]

# Scale the features
county_features_scaled = scaler.transform([county_features.values])[0]

# Predict
risk_prob = model.predict_proba([county_features_scaled])[0][1]
risk_label = "High Risk" if risk_prob > 0.5 else "Stable"

# ---- Display risk score ----
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Abandonment Risk", f"{risk_prob*100:.0f}%")

with col2:
    st.metric("Risk Category", risk_label)

with col3:
    acreage_pct = county_data["acreage_change_pct"]
    st.metric("5-yr Acreage Change", f"{acreage_pct:.1f}%")

st.divider()

# ---- SHAP explanation ----
st.subheader("Drivers of Risk")
st.write("Below are the factors pushing this county's risk score, ranked by impact.")

# Compute SHAP for this specific county
county_shap = explainer.shap_values(np.array([county_features_scaled]))[0]

# Create driver DataFrame
drivers = pd.DataFrame({
    "Driver": county_features.index,
    "Value": county_features.values,
    "SHAP Impact": county_shap,
    "Direction": ["Increases Risk" if x < 0 else "Decreases Risk" for x in county_shap]
})
drivers["Abs Impact"] = np.abs(drivers["SHAP Impact"])
drivers = drivers.sort_values("Abs Impact", ascending=False)

# Display top 4 drivers
for idx, row in drivers.head(4).iterrows():
    if row["SHAP Impact"] < 0:
        impact_text = f"Increases risk by {abs(row['SHAP Impact']):.2f}"
        color = "red"
    else:
        impact_text = f"Decreases risk by {row['SHAP Impact']:.2f}"
        color = "green"
    
    st.write(f"**{row['Driver']}**")
    st.write(f"Current value: {row['Value']:.2f} — {impact_text}")
    st.divider()

# ---- Risk assessment ----
st.subheader("Assessment")

if risk_prob > 0.5:
    st.error(f"High Risk. {county} County shows multiple signals of agricultural decline.")
    st.write("""
    Recommended interventions:
    - Transition to low-pesticide precision agriculture (reduce input costs 30-40%)
    - Enroll in USDA NRCS conservation programs
    - Diversify crop mix and explore alternative crops (hemp, regenerative grazing)
    """)
else:
    st.success(f"Stable. {county} County farmland is currently trending well.")

st.divider()

# ---- Model info ----
with st.expander("Model Details"):
    st.write("""
    **Data Source:** USDA NASS Census of Agriculture (2017 vs 2022)
    
    **Model:** Logistic Regression (4 features)
    
    **Features:**
    - Acreage change (% 2017→2022)
    - Farm count change (% 2017→2022)
    - Average farm size change (% 2017→2022)
    - Consolidation ratio (farm decline vs acreage decline)
    
    **Performance:** 77% accuracy, 57% recall on high-risk counties
    
    **Explainability:** SHAP (SHapley Additive exPlanations) shows feature-level 
    contribution to each prediction. Negative SHAP values increase abandonment risk; 
    positive values decrease it.
    """)

# ---- All counties table ----
with st.expander("View all Kentucky counties"):
    display_df = features_df.copy()
    display_df = display_df.round(2)
    display_df = display_df.sort_values("acreage_change_pct")
    st.dataframe(display_df, use_container_width=True)

st.caption("Viridis v2.0 — Girls in STEM Global Hackathon & DSH Pitch Competition")
