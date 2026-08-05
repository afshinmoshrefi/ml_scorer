"""Diagnose ensemble diversity and weak segments on WF predictions (v4)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

RES = "C:/seasonals/ml_scorer/results"

def auc_safe(y, p):
    y = np.asarray(y)
    if len(y) < 100 or y.min() == y.max():
        return float("nan")
    return roc_auc_score(y, p)

# ---------------- 10_30 tier ----------------
cols = ["val_year", "predicted", "pred_lgb", "pred_xgb", "pred_cb",
        "actual_return", "hit_target", "daysOut", "direction"]
df = pd.read_parquet(f"{RES}/wf_predictions_sr_v4.parquet", columns=cols)
print(f"10_30 rows: {len(df):,}")

# 1. DIVERSITY: pairwise Pearson corr on 2M fixed-seed sample
samp = df.sample(n=2_000_000, random_state=42)
corr = samp[["pred_lgb", "pred_xgb", "pred_cb"]].corr(method="pearson")
print("\n=== Pairwise Pearson corr (2M sample, seed 42) ===")
print(corr.round(4))
lgb_xgb = corr.loc["pred_lgb", "pred_xgb"]
lgb_cb = corr.loc["pred_lgb", "pred_cb"]
xgb_cb = corr.loc["pred_xgb", "pred_cb"]
print(f"RESULT pairwise_corr lgb_xgb={lgb_xgb:.4f} lgb_cb={lgb_cb:.4f} xgb_cb={xgb_cb:.4f}")

# 2. AUC of ensemble
print("\n=== AUC overall ===")
overall = auc_safe(df["hit_target"], df["predicted"])
print(f"overall: {overall:.4f}")

print("\n=== AUC by val_year ===")
by_year = {}
for y, g in df.groupby("val_year"):
    by_year[int(y)] = auc_safe(g["hit_target"], g["predicted"])
    print(f"{y}: {by_year[int(y)]:.4f}  (n={len(g):,}, base WR={g['hit_target'].mean():.3f})")

print("\n=== AUC by direction ===")
by_dir = {}
for d, g in df.groupby("direction"):
    by_dir[d] = auc_safe(g["hit_target"], g["predicted"])
    print(f"{d}: {by_dir[d]:.4f}  (n={len(g):,}, base WR={g['hit_target'].mean():.3f})")

print("\n=== AUC by daysOut bucket ===")
buckets = pd.cut(df["daysOut"], bins=[9, 15, 22, 30], labels=["10-15", "16-22", "23-30"])
by_bucket = {}
for b, g in df.groupby(buckets, observed=True):
    by_bucket[str(b)] = auc_safe(g["hit_target"], g["predicted"])
    print(f"{b}: {by_bucket[str(b)]:.4f}  (n={len(g):,})")

# year x direction segments for weak-segment hunt
print("\n=== AUC by year x direction (10_30) ===")
seg_rows = []
for (y, d), g in df.groupby(["val_year", "direction"]):
    a = auc_safe(g["hit_target"], g["predicted"])
    seg_rows.append(("10_30", int(y), d, a, len(g)))
    print(f"{y} {d}: {a:.4f} (n={len(g):,})")

print("\n=== AUC by year x daysOut bucket (10_30) ===")
for (y, b), g in df.groupby(["val_year", buckets], observed=True):
    a = auc_safe(g["hit_target"], g["predicted"])
    print(f"{y} {b}: {a:.4f} (n={len(g):,})")

# 4. DISAGREEMENT TEST
print("\n=== Disagreement test ===")
for c in ["pred_lgb", "pred_xgb", "pred_cb"]:
    grp = df.groupby("val_year")[c]
    df[c + "_z"] = (df[c] - grp.transform("mean")) / grp.transform("std")
zc = ["pred_lgb_z", "pred_xgb_z", "pred_cb_z"]
df["disagreement"] = df[zc].std(axis=1, ddof=1)

# Spearman: disagreement vs |hit_target - within-year percentile rank of predicted|
df["pred_rank"] = df.groupby("val_year")["predicted"].rank(pct=True)
df["cal_err"] = (df["hit_target"] - df["pred_rank"]).abs()
ds = df.sample(n=2_000_000, random_state=42)
rho, _ = spearmanr(ds["disagreement"], ds["cal_err"])
print(f"Spearman(disagreement, |hit - rank(pred)|) on 2M sample: {rho:.4f}")

# top decile of predicted within each year, split at median disagreement
df["is_top"] = df.groupby("val_year")["predicted"].rank(pct=True) >= 0.90
top = df[df["is_top"]].copy()
med = top.groupby("val_year")["disagreement"].transform("median")
low = top[top["disagreement"] <= med]
high = top[top["disagreement"] > med]
wr_low, wr_high = low["hit_target"].mean(), high["hit_target"].mean()
print(f"ML_90 bucket n={len(top):,}; low-disagree WR={wr_low:.4f} (n={len(low):,}), "
      f"high-disagree WR={wr_high:.4f} (n={len(high):,}), diff={(wr_low-wr_high)*100:.2f}pp")
ret_low, ret_high = low["actual_return"].mean(), high["actual_return"].mean()
print(f"ML_90 avg actual_return: low-disagree={ret_low:.3f}%, high-disagree={ret_high:.3f}%")
# per-year breakdown
print("per-year ML_90 WR low vs high disagreement:")
for y, g in top.groupby("val_year"):
    m = g["disagreement"].median()
    lo = g[g["disagreement"] <= m]["hit_target"].mean()
    hi = g[g["disagreement"] > m]["hit_target"].mean()
    print(f"  {y}: low={lo:.3f} high={hi:.3f} diff={(lo-hi)*100:+.1f}pp")

del df, samp, ds, top, low, high

# ---------------- 31_60 / 61_90: AUC by direction ----------------
for tier, fn in [("31_60", "wf_predictions_sr_31_60_v4.parquet"),
                 ("61_90", "wf_predictions_sr_61_90_v4.parquet")]:
    d2 = pd.read_parquet(f"{RES}/{fn}",
                         columns=["val_year", "predicted", "hit_target", "direction"])
    print(f"\n=== {tier} (n={len(d2):,}) ===")
    print(f"overall AUC: {auc_safe(d2['hit_target'], d2['predicted']):.4f}")
    for d, g in d2.groupby("direction"):
        print(f"{tier} dir={d}: AUC={auc_safe(g['hit_target'], g['predicted']):.4f} (n={len(g):,})")
    # year x direction for weak segments
    for (y, d), g in d2.groupby(["val_year", "direction"]):
        print(f"  {tier} {y} {d}: {auc_safe(g['hit_target'], g['predicted']):.4f} (n={len(g):,})")
    del d2

print("\nDONE")
