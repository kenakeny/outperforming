"""
vol_absolute_vs_relative.py — isolate the volatility signal: absolute vs
peer-relative, direction and predictive value. Same next-5d peer-relative
tercile label + 2026 holdout as train_5d.py.

Features (all trailing / leak-free, OHLCV only):
  ABSOLUTE:
    vol_20d, vol_60d          realized std of daily returns
    hl_range_20d              mean intraday (High-Low)/Close
  PEER-RELATIVE (within category, leave-one-out so a fund never benchmarks itself):
    vol_20d_rel, vol_60d_rel  fund vol minus its category-peer mean vol
    vol_20d_catrank           fund vol's cross-sectional pct-rank within category (0..1)

Outputs: direction (mean forward peer-rel return by feature decile) + a CatBoost
holdout using ONLY these volatility features.

Run:  python vol_absolute_vs_relative.py
"""
import pathlib
import yaml
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score

HORIZON, SKIP, EMBARGO, MIN_GROUP, TEST_YEAR = 5, 0, 5, 3, 2026
TASK_TYPE = "GPU"

ROOT = pathlib.Path(__file__).resolve().parent
cfg  = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RAW  = ROOT / cfg["paths"]["raw"]
PROC = ROOT / cfg["paths"]["processed"]

md = pd.read_parquet(RAW / "market_data.parquet")
md.index = pd.to_datetime(md.index); md.index.name = "date"
close, high, low = md["Close"], md["High"], md["Low"]
for f in (close, high, low): f.columns.name = "ticker"
trading_index = close.index

# restrict to real categories (>= MIN_GROUP peers, not Uncategorized)
metadata = pd.read_parquet(RAW / "metadata.parquet")
cat = metadata["category"].reindex(close.columns)
counts = cat.value_counts()
keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
kept = cat[cat.isin(keep)].index
close, high, low, cat = close[kept], high[kept], low[kept], cat[kept]
print("universe:", close.shape, "| categories:", cat.nunique())

# ---------------- absolute volatility features ----------------
r1 = close.pct_change(fill_method=None)
vol_20d = r1.rolling(20).std()
vol_60d = r1.rolling(60).std()
hl_range_20d = ((high - low) / close).rolling(20).mean()

# ---------------- peer-relative versions (leave-one-out within category, same day) ----------------
def loo_peer_mean(x):
    """each cell minus itself: category mean over the OTHER funds that day."""
    s = x.T.groupby(cat).transform("sum").T
    n = x.notna().T.groupby(cat).transform("sum").T
    return (s - x.fillna(0)) / (n - x.notna().astype(int)).replace(0, np.nan)

def cat_pct_rank(x):
    """cross-sectional percentile rank of x within its category, per day (0..1)."""
    return x.T.groupby(cat).rank(pct=True).T

vol_20d_rel = vol_20d - loo_peer_mean(vol_20d)
vol_60d_rel = vol_60d - loo_peer_mean(vol_60d)
vol_20d_catrank = cat_pct_rank(vol_20d)

feat = {
    "vol_20d": vol_20d, "vol_60d": vol_60d, "hl_range_20d": hl_range_20d,
    "vol_20d_rel": vol_20d_rel, "vol_60d_rel": vol_60d_rel, "vol_20d_catrank": vol_20d_catrank,
}
ABS  = ["vol_20d", "vol_60d", "hl_range_20d"]
REL  = ["vol_20d_rel", "vol_60d_rel", "vol_20d_catrank"]
FEATURES = ABS + REL

def _stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s
X = pd.DataFrame({k: _stack(v) for k, v in feat.items()})

# ---------------- label + continuous peer-relative return (for the direction check) ----------------
fwd = close.shift(-HORIZON) / close.shift(-SKIP) - 1.0
grp_sum = fwd.T.groupby(cat).transform("sum").T
grp_n   = fwd.notna().T.groupby(cat).transform("sum").T
peer_mean = (grp_sum - fwd.fillna(0)) / (grp_n - fwd.notna().astype(int)).replace(0, np.nan)
rel = fwd - peer_mean
rel_l = _stack(rel)
target = (rel_l.dropna().groupby(level="date")
          .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=[0, 1, 2])).astype("int8"))

d = X.join(pd.concat({"target": target, "rel": rel_l}, axis=1), how="inner")
d = d.dropna(subset=["target"])
d["target"] = d["target"].astype("int8")

# ---------------- DIRECTION: mean forward peer-rel return by decile of each vol feature ----------------
print("\n=== direction: mean next-5d PEER-RELATIVE return (bps) by feature decile ===")
print("    decile 1 = lowest volatility, decile 10 = highest\n")
dd = d.dropna(subset=["rel"])
rows = {}
for f in FEATURES:
    sub = dd[[f, "rel"]].dropna()
    dec = pd.qcut(sub[f].rank(method="first"), 10, labels=False) + 1
    rows[f] = (sub.groupby(dec)["rel"].mean() * 1e4).round(1)   # bps
table = pd.DataFrame(rows)
table.index.name = "decile"
print(table.to_string())
# monotonic spread (D10 - D1): + => high-vol outperforms, - => low-vol outperforms
spread = (table.iloc[-1] - table.iloc[0]).round(1)
print("\nD10 - D1 spread (bps)  [+ high-vol wins, - low-vol wins]:")
print(spread.to_string())

# ---------------- CatBoost holdout: volatility features only ----------------
date_index = d.index.get_level_values("date")
test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
cut = trading_index[max(trading_index.searchsorted(test_start) - EMBARGO, 0)]
tr_mask, te_mask = date_index <= cut, date_index >= test_start
tr_dates = np.sort(date_index[tr_mask].unique())
val_start = tr_dates[int(len(tr_dates) * 0.90)]
val_cut = trading_index[max(trading_index.searchsorted(val_start) - EMBARGO, 0)]
fit_mask = tr_mask & (date_index <= val_cut)
val_mask = tr_mask & (date_index >= val_start)

params = dict(iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=3,
              loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
              random_seed=42, task_type=TASK_TYPE, verbose=0)

def run(cols, tag):
    m = CatBoostClassifier(early_stopping_rounds=100, use_best_model=True, **params)
    m.fit(d.loc[fit_mask, cols], d.loc[fit_mask, "target"],
          eval_set=(d.loc[val_mask, cols], d.loc[val_mask, "target"]))
    yte = d.loc[te_mask, "target"]
    pr  = m.predict(d.loc[te_mask, cols]).astype(int).ravel()
    f1  = f1_score(yte, pr, average="macro"); acc = accuracy_score(yte, pr)
    proba = m.predict_proba(d.loc[te_mask, cols])[:, 2]
    icd = pd.DataFrame({"s": proba, "rel": d.loc[te_mask, "rel"]}, index=yte.index)
    ic = icd.groupby(level="date").apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean()
    imp = pd.Series(m.get_feature_importance(), index=cols).sort_values(ascending=False)
    print(f"\n[{tag}]  macro F1={f1:.4f}  acc={acc:.4f}  rankIC={ic:.4f}")
    print(imp.round(2).to_string())
    return f1, ic

print("\n=== CatBoost holdout 2026 (volatility features only) ===")
run(ABS,      "ABSOLUTE vol only")
run(REL,      "PEER-RELATIVE vol only")
run(FEATURES, "ABSOLUTE + PEER-RELATIVE")
