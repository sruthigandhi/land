import pandas as pd
df = pd.read_csv("data/ky_features_final.csv", index_col=0)
thr = df.loc[df.high_risk == 1, "acreage_change_pct"].max()
rule = (df.acreage_change_pct <= thr).astype(int)
print(f"Label threshold (acreage change %): {thr:.2f}")
print(f"One-line rule 'acreage_change_pct <= threshold' reproduces high_risk for {(rule == df.high_risk).mean()*100:.1f}% of counties")
print(f"High-risk share: {df.high_risk.mean()*100:.1f}%")
