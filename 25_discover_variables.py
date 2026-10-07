import os, requests
from dotenv import load_dotenv
load_dotenv()
KEY = os.getenv("NASS_API_KEY")
r = requests.get("https://quickstats.nass.usda.gov/api/get_param_values/",
                 params=dict(key=KEY, param="short_desc", source_desc="CENSUS",
                             state_alpha="KY", agg_level_desc="COUNTY", year=2017), timeout=120)
print("HTTP", r.status_code)
vals = r.json().get("short_desc", []) if r.status_code == 200 else []
open("data/census_short_desc_2017.txt", "w").write("\n".join(vals))
print(len(vals), "variables available at county level in 2017\n")
for k in ["OPERATORS, PRINCIPAL", " AGE", "INCOME, NET CASH", "SALES, MEASURED IN $",
          "GOVT PROGRAMS", "CROPLAND - ACRES", "RENT", "TENURE"]:
    hits = [v for v in vals if k in v][:12]
    print(f"== contains '{k}' ({len(hits)} shown) ==")
    for h in hits:
        print("  ", h)
