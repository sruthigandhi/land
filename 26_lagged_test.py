import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

raw = pd.read_csv("data/panel_raw.csv")
w = raw.pivot_table(index="county", columns=["item", "year"], values="value")
w.columns = [f"{i}_{y}" for i, y in w.columns]
pct = lambda a, b: ((a - b) / b * 100).replace([np.inf, -np.inf], np.nan)

frames = []
for o in [2012, 2017]:
    d = pd.DataFrame(index=w.index)
    d["origin"] = o
    d["overlap_prior"] = pct(w[f"acres_{o}"], w[f"acres_{o-5}"])      # shares an endpoint with the outcome
    d["lagged_prior"] = pct(w[f"acres_{o-5}"], w[f"acres_{o-10}"])    # no shared endpoint
    d["next"] = pct(w[f"acres_{o+5}"], w[f"acres_{o}"])
    frames.append(d)
p = pd.concat(frames).dropna()
tr, te = p[p.origin == 2012], p[p.origin == 2017]

print("Spearman rho with next-period acreage change")
for f in ["overlap_prior", "lagged_prior"]:
    for o, d in [(2012, tr), (2017, te)]:
        rho, pv = spearmanr(d[f], d["next"])
        print(f"  {f:14s} origin {o}: rho {rho:+.3f}  p {pv:.3f}")

def boot(y, s, n=1000):
    rng, out = np.random.default_rng(0), []
    y, s = np.asarray(y), np.asarray(s)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])

thr = tr["next"].quantile(0.25)
labels = {f"fixed cutoff ({thr:.1f}%)": te["next"] <= thr,
          "test-period bottom quartile": te["next"] <= te["next"].quantile(0.25)}
print("\nOut-of-time AUC on 2017 origin (direction learned on 2012 only)")
for f in ["overlap_prior", "lagged_prior"]:
    sign = np.sign(spearmanr(tr[f], tr["next"])[0])
    score = -sign * te[f]
    for name, y in labels.items():
        y = y.astype(int)
        lo, hi = boot(y, score)
        print(f"  {f:14s} | {name:30s} AUC {roc_auc_score(y, score):.3f} (95% CI {lo:.2f}-{hi:.2f})")
