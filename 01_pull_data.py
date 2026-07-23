"""
Pulls real Kentucky county-level farmland data from the USDA NASS QuickStats API.
Get a free API key first: https://quickstats.nass.usda.gov/api (instant, just email)

Run this in Positron's terminal or Run button. Saves data/ky_farmland_raw.csv
"""

import requests
import pandas as pd
import os

API_KEY = "B67AFA2E-5947-3EED-A7DF-3CEF591D2B56"  # Your key
BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

def pull_land_in_farms(year):
    """Pulls 'land in farms' acreage by KY county for a specific year."""
    params = {
        "key": API_KEY,
        "source_desc": "CENSUS",
        "sector_desc": "ECONOMICS",
        "group_desc": "FARMS & LAND & ASSETS",
        "commodity_desc": "AG LAND",
        "statisticcat_desc": "AREA",
        "state_alpha": "KY",
        "agg_level_desc": "COUNTY",
        "year": year,  # <-- CRITICAL: query one year at a time
        "format": "JSON",
    }
    print(f"Requesting {year} data from NASS API...")
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Create data folder
    os.makedirs("data", exist_ok=True)
    
    print("Pulling land-in-farms data for Census years 2017 and 2022...\n")
    
    all_data = []
    
    for year in [2017, 2022]:
        try:
            land_df = pull_land_in_farms(year)
            print(f"✅ {year}: Got {len(land_df)} rows")
            all_data.append(land_df)
        except Exception as e:
            print(f"❌ {year}: Error - {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv("data/ky_land_in_farms_raw.csv", index=False)
        print(f"\n✅ SUCCESS: Saved {len(combined_df)} total rows to data/ky_land_in_farms_raw.csv")
        
        # Show a sample
        print("\nSample of data:")
        print(combined_df[["county_name", "year", "short_desc", "Value"]].head(10))
    else:
        print("\n❌ FAILED: No data retrieved.")
        