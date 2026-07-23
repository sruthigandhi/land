"""
Scenario simulation engine.

User adjusts inputs (commodity price, rainfall, farm consolidation)
and see how risk changes in real-time.
"""

import pandas as pd
import numpy as np
import pickle

# Load model + scaler
with open("model_rf_v3.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_rf_v3.pkl", "rb") as f:
    scaler = pickle.load(f)

features = pd.read_csv("data/ky_features_final.csv", index_col=0)

def predict_with_scenario(county, scenario_adjustments):
    """
    Predict risk under a scenario.
    
    scenario_adjustments: dict like {
        'commodity_price_change': 20,  # % change
        'rainfall_change': -15,
        'consolidation_rate': 5,
        'conservation_investment': 50  # USDA program enrollment
    }
    """
    base = features.loc[county].copy()
    
    # Apply scenario adjustments as deltas to features
    if 'commodity_price_change' in scenario_adjustments:
        # Higher prices → better margins → less abandonment
        base['acreage_change_pct'] -= scenario_adjustments['commodity_price_change'] * 0.1
    
    if 'consolidation_rate' in scenario_adjustments:
        # More consolidation → worse for small farms
        base['consolidation_ratio'] += scenario_adjustments['consolidation_rate'] * 0.5
    
    if 'conservation_investment' in scenario_adjustments:
        # Conservation programs reduce risk
        base['acreage_change_pct'] += scenario_adjustments['conservation_investment'] * 0.05
    
    if 'rainfall_change' in scenario_adjustments:
        # Drought/flood affects profitability
        base['avg_farm_size_change_pct'] -= scenario_adjustments['rainfall_change'] * 0.2
    
    # Predict
    features_cols = ["acreage_change_pct", "farm_count_change_pct", "avg_farm_size_change_pct", "consolidation_ratio"]
    scaled = scaler.transform([base[features_cols].values])[0]
    risk_prob = model.predict_proba([scaled])[0][1]
    
    return risk_prob

# Test scenarios
print("Scenario Analysis: Fulton County\n")
print("=" * 60)

base_risk = predict_with_scenario("FULTON", {})
print(f"Current risk: {base_risk*100:.0f}%\n")

scenarios = {
    "Soybean prices +20%": {'commodity_price_change': 20},
    "Soybean prices -20%": {'commodity_price_change': -20},
    "Severe drought (-30% rainfall)": {'rainfall_change': -30},
    "NRCS conservation enrollment": {'conservation_investment': 50},
    "Rapid farm consolidation": {'consolidation_rate': 15},
}

for scenario_name, adjustments in scenarios.items():
    new_risk = predict_with_scenario("FULTON", adjustments)
    delta = (new_risk - base_risk) * 100
    direction = "↑" if delta > 0 else "↓"
    
    print(f"{scenario_name}")
    print(f"  Risk: {new_risk*100:.0f}% ({direction} {abs(delta):.0f}pp)\n")

print("=" * 60)
print("\n✅ Scenario engine working!")
print("Next: Integrate into Streamlit app")
