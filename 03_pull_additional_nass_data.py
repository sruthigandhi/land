"""
Pull additional NASS data for feature engineering:
- Farm count by county
- Farm income
- Operator demographics

These give us the 'drivers' of abandonment risk.
"""

import requests
import pandas as pd
import os

API_KEY = "B67AFA2E-5947-3EED-A7DF-3CEF591D2B56"
BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

def pull_nass(commodity_desc, statisticcat_desc, year):
    """Generic NASS puller."""
    params = {
        "key": API_KEY,
        "source_desc": "CENSUS",
        "sector_desc": "ECONOMICS",
        "commodity_desc": commodity_desc,
        "statisticcat_desc": statisticcat_desc,
        "state_alpha": "KY",
        "agg_level_desc": "COUNTY",
        "year": year,
        "format": "JSON",
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()["data"]
        return pd.DataFrame(data)
    except Exception as e:
        print(f"  Error pulling {commodity_desc}/{statisticcat_desc}: {e}")
        return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Pulling farm count data (2017, 2022)...\n")
    farm_data = []
    for year in [2017, 2022]:
        df = pull_nass("FARM OPERATIONS", "OPERATIONS", year)
        if df is not None:
            print(f"  {year}: {len(df)} rows")
            farm_data.append(df)

    if farm_data:
        farms_combined = pd.concat(farm_data, ignore_index=True)
        farms_combined.to_csv("data/ky_farm_operations_raw.csv", index=False)
        print(f"✅ Saved farm operations to data/ky_farm_operations_raw.csv\n")

    print("Pulling farm income data (2017, 2022)...\n")
    income_data = []
    for year in [2017, 2022]:
        # Try "VALUE OF PRODUCTION" first
        df = pull_nass("FARM OPERATIONS", "VALUE OF PRODUCTION", year)
        if df is not None and len(df) > 0:
            print(f"  {year}: {len(df)} rows")
            income_data.append(df)

    if income_data:
        income_combined = pd.concat(income_data, ignore_index=True)
        income_combined.to_csv("data/ky_farm_income_raw.csv", index=False)
        print(f"✅ Saved farm income to data/ky_farm_income_raw.csv\n")
    else:
        print("  Note: Income data may not be available via NASS CENSUS queries.")
        print("  We'll engineer what we can from farm count + acreage.\n")

    print("Data pull complete. Next: 04_engineer_features.py")
