import os, time, requests, pandas as pd
from dotenv import load_dotenv
load_dotenv()
KEY = os.getenv("NASS_API_KEY")
URL = "https://quickstats.nass.usda.gov/api/api_GET/"
VARS = {
 "prod_age_avg":   "PRODUCERS - AGE, AVG, MEASURED IN YEARS",
 "oper_age_avg":   "OPERATORS, PRINCIPAL - AGE, AVG, MEASURED IN YEARS",
 "prod_n":         "PRODUCERS - NUMBER OF PRODUCERS",
 "prod_65_74":     "PRODUCERS, AGE 65 TO 74 - NUMBER OF PRODUCERS",
 "prod_75plus":    "PRODUCERS, AGE GE 75 - NUMBER OF PRODUCERS",
 "prod_le35":      "PRODUCERS, AGE LE 35 - NUMBER OF PRODUCERS",
 "netinc_per_op":  "INCOME, NET CASH FARM, OF OPERATIONS - NET INCOME, MEASURED IN $ / OPERATION",
 "ops_loss":       "INCOME, NET CASH FARM, OF OPERATIONS - OPERATIONS WITH LOSS",
 "ops_netinc":     "INCOME, NET CASH FARM, OF OPERATIONS - OPERATIONS WITH NET INCOME",
 "sales_per_op":   "COMMODITY TOTALS - SALES, MEASURED IN $ / OPERATION",
 "govt_per_op":    "GOVT PROGRAMS, FEDERAL - RECEIPTS, MEASURED IN $ / OPERATION",
 "cropland_acres": "AG LAND, CROPLAND - ACRES",
 "rented_acres":   "AG LAND, RENTED FROM OTHERS, IN FARMS - ACRES",
}
rows = []
for name, desc in VARS.items():
    for year in [2002, 2007, 2012, 2017, 2022]:
        params = dict(key=KEY, source_desc="CENSUS", state_alpha="KY", agg_level_desc="COUNTY",
                      year=year, short_desc=desc, domain_desc="TOTAL", format="JSON")
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            for rec in r.json().get("data", []):
                rows.append({"var": name, "year": year, "county": rec["county_name"], "value": rec["Value"]})
        time.sleep(0.2)
df = pd.DataFrame(rows)
df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(",", ""), errors="coerce")
df = df[~df.county.str.contains("OTHER", case=False)].drop_duplicates(["var", "year", "county"])
df.to_csv("data/extra_raw.csv", index=False)
print("Counties with a numeric value, by variable and year (max 120):")
print(df.dropna(subset=["value"]).pivot_table(index="var", columns="year", values="county",
      aggfunc="nunique").reindex(list(VARS)).fillna(0).astype(int))
