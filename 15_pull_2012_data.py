"""
Pull 2012 Census data — needed to build a leak-free temporal dataset.
We'll use 2012 features to predict 2017 outcomes, then validate
by using 2017 features to predict 2022 outcomes.
"""

import requests
import pandas as pd

import os

api_key = os.getenv("REMOVED_NASS_KEY")
if not api_key:
    raise ValueError("NASS_API_KEY not set. Set it in your shell: export NASS_API_KEY='...'")BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

def pull_nass(commodity_desc, statisticcat_desc, group_desc, year):
    params = {
        "key": API_KEY,
        "source_desc": "CENSUS",
        "sector_desc": "ECONOMICS",
        "group_desc": group_desc,
        "commodity_desc": commodity_desc,
        "statisticcat_desc": statisticcat_desc,
        "state_alpha": "KY",
        "agg_level_desc": "COUNTY",
        "year": year,
        "format": "JSON",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Pulling 2012 land-in-farms data...")
    land_2012 = pull_nass("AG LAND", "AREA", "FARMS & LAND & ASSETS", 2012)
    print(f"  Got {len(land_2012)} rows")
    land_2012.to_csv("data/ky_land_2012_raw.csv", index=False)

    print("Pulling 2012 farm operations data...")
    farms_2012 = pull_nass("FARM OPERATIONS", "OPERATIONS", "FARMS & LAND & ASSETS", 2012)
    print(f"  Got {len(farms_2012)} rows")
    farms_2012.to_csv("data/ky_farms_2012_raw.csv", index=False)

    print("\n✅ Saved 2012 data. Now we have 3 time points: 2012, 2017, 2022")
    