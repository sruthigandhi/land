import os, requests, pandas as pd
from dotenv import load_dotenv
load_dotenv()
KEY = os.getenv("NASS_API_KEY")
URL = "https://quickstats.nass.usda.gov/api/api_GET/"
ITEMS = {"acres": "FARM OPERATIONS - ACRES OPERATED",
         "farms": "FARM OPERATIONS - NUMBER OF OPERATIONS"}
rows = []
for item, desc in ITEMS.items():
    params = dict(key=KEY, source_desc="CENSUS", state_alpha="KY", agg_level_desc="COUNTY",
                  year=2002, short_desc=desc, domain_desc="TOTAL", format="JSON")
    r = requests.get(URL, params=params, timeout=60)
    if r.status_code != 200:
        print(f"FAILED 2002 {item}: HTTP {r.status_code} {r.text[:150]}")
        continue
    for rec in r.json().get("data", []):
        rows.append({"year": 2002, "item": item, "county": rec["county_name"], "value": rec["Value"]})
new = pd.DataFrame(rows)
if new.empty:
    print("No 2002 rows returned")
else:
    new["value"] = pd.to_numeric(new["value"].str.replace(",", ""), errors="coerce")
    new = new[~new.county.str.contains("OTHER", case=False)].drop_duplicates(["year", "item", "county"])
    raw = pd.read_csv("data/panel_raw.csv")
    raw = raw[raw.year != 2002]
    pd.concat([raw, new]).to_csv("data/panel_raw.csv", index=False)
    print("Expect roughly 13.8M acres and 86,000 farms statewide in 2002")
    print(new.groupby("item").agg(counties=("county", "nunique"),
          missing=("value", lambda s: s.isna().sum()), statewide_total=("value", "sum")))
