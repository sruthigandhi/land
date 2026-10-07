import pandas as pd
import numpy as np

INPUT = "data/panel_raw.csv"
OUTPUT = "data/targets.csv"

raw = pd.read_csv(INPUT)

# Expected columns:
# county, year, item, value

w = raw.pivot_table(
    index="county",
    columns=["item", "year"],
    values="value",
    aggfunc="first"
)

w.columns = [f"{item}_{year}" for item, year in w.columns]

rows = []

for origin in [2012, 2017]:

    future = origin + 5

    current_col = f"acres_{origin}"
    future_col = f"acres_{future}"

    if current_col not in w.columns or future_col not in w.columns:
        raise ValueError(
            f"Missing acreage columns for {origin}->{future}: "
            f"{current_col}, {future_col}"
        )

    d = pd.DataFrame(index=w.index)

    d["county"] = d.index
    d["origin_year"] = origin
    d["future_year"] = future

    d["future_acres"] = w[future_col]
    d["origin_acres"] = w[current_col]

    d["future_acres_change_pct"] = (
        (w[future_col] - w[current_col])
        / w[current_col]
        * 100
    )

    rows.append(d.reset_index(drop=True))

targets = pd.concat(rows, ignore_index=True)

# Define high decline separately WITHIN each future period.
targets["high_decline"] = (
    targets
    .groupby("origin_year")["future_acres_change_pct"]
    .transform(lambda x: x <= x.quantile(0.25))
    .astype(int)
)

targets.to_csv(OUTPUT, index=False)

print("\nTARGET SUMMARY")
print(targets.groupby("origin_year").agg(
    counties=("county", "count"),
    mean_change=("future_acres_change_pct", "mean"),
    median_change=("future_acres_change_pct", "median"),
    high_decline=("high_decline", "sum")
))

print("\nSAMPLE")
print(targets.head(10).to_string(index=False))
