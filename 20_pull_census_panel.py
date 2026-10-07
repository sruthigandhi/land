import os, requests, pandas as pd
from dotenv import load_dotenv
load_dotenv()
KEY = os.getenv("NASS_API_KEY")
URL = "https://quickstats.nass.usda.gov/api/api_GET/"
ITEMS = {"acres": "FARM OPERATIONS - ACRES OPERATED",
         "farms": "FARM OPERATIONS - NUMBER OF OPERATIONS"}
rows = []
for year in [2007, 2012, 2017, 2022]:
    for item, desc in ITEMS.items():
        params = dict(key=KEY, source_desc="CENSUS", state_alpha="KY", agg_level_desc="COUNTY",
                      year=year, short_desc=desc, domain_desc="TOTAL", format="JSON")
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code != 200:
            print(f"FAILED {year} {item}: HTTP {r.status_code} {r.text[:150]}")
            continue
        for rec in r.json().get("data", []):
            rows.append({"year": year, "item": item, "county": rec["county_name"],
                         "value": rec["Value"]})
df = pd.DataFrame(rows)
df["value"] = pd.to_numeric(df["value"].str.replace(",", ""), errors="coerce")
df = df[~df.county.str.contains("OTHER", case=False)]
df = df.drop_duplicates(["year", "item", "county"])
df.to_csv("data/panel_raw.csv", index=False)
print(df.groupby(["item", "year"]).agg(counties=("county", "nunique"),
      missing=("value", lambda s: s.isna().sum()), statewide_total=("value", "sum")))
