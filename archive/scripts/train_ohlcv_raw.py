"""
train_ohlcv_raw.py — the most basic possible CatBoost: the FIVE raw yfinance
columns as features, nothing derived. Open/High/Low/Close/Volume at day t -> the
next-5-day peer-relative tercile label. Same label + holdout as train_5d.py so
it's directly comparable to the engineered models.

Caveat by construction: these are absolute levels (price scale differs wildly
across funds), so there's little cross-sectional signal here — this is the
floor, not a serious model.

Run:  python train_ohlcv_raw.py
"""
import pathlib
import yaml
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

HORIZON, SKIP, EMBARGO, MIN_GROUP, TEST_YEAR = 5, 0, 5, 3, 2026
TASK_TYPE = "GPU"

ROOT = pathlib.Path(__file__).resolve().parent
cfg  = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RAW  = ROOT / cfg["paths"]["raw"]
PROC = ROOT / cfg["paths"]["processed"]

md = pd.read_parquet(RAW / "market_data.parquet")
md.index = pd.to_datetime(md.index); md.index.name = "date"
close = md["Close"]; close.columns.name = "ticker"
trading_index = close.index
print("raw OHLCV:", close.shape, "|", trading_index.min().date(), "->", trading_index.max().date())

def _stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s

# --- the ONLY features: the five raw OHLCV columns at day t ---
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
X = pd.DataFrame({f: _stack(md[f]) for f in FEATURES})

# --- label: next-5d peer-relative tercile (identical to train_5d.py) ---
metadata = pd.read_parquet(RAW / "metadata.parquet")
cat = metadata["category"].reindex(close.columns)
counts = cat.value_counts()
keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
kept_cols = cat[cat.isin(keep)].index
px, cat = close[kept_cols], cat[kept_cols]

fwd = px.shift(-HORIZON) / px.shift(-SKIP) - 1.0
grp_sum = fwd.T.groupby(cat).transform("sum").T
grp_n   = fwd.notna().T.groupby(cat).transform("sum").T
peer_mean = (grp_sum - fwd.fillna(0)) / (grp_n - fwd.notna().astype(int)).replace(0, np.nan)
rel = fwd - peer_mean
rel_l = _stack(rel)
target = (rel_l.dropna().groupby(level="date")
          .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=[0, 1, 2])).astype("int8"))
lbl = target.rename("target").to_frame()

d = X.join(lbl, how="inner").dropna(subset=["target"] + FEATURES)
d["target"] = d["target"].astype("int8")
print("labeled dataset:", d.shape,
      "| class balance:", dict(d["target"].value_counts(normalize=True).round(2)))

date_index = d.index.get_level_values("date")
test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
cut = trading_index[max(trading_index.searchsorted(test_start) - EMBARGO, 0)]
tr_mask, te_mask = date_index <= cut, date_index >= test_start
tr_dates = np.sort(date_index[tr_mask].unique())
val_start = tr_dates[int(len(tr_dates) * 0.90)]
val_cut = trading_index[max(trading_index.searchsorted(val_start) - EMBARGO, 0)]
fit_mask = tr_mask & (date_index <= val_cut)
val_mask = tr_mask & (date_index >= val_start)

params = dict(iterations=2000, learning_rate=0.04, depth=7, l2_leaf_reg=3,
              loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
              random_seed=42, task_type=TASK_TYPE, verbose=200)

print("\ntraining raw-OHLCV holdout model ...")
model = CatBoostClassifier(early_stopping_rounds=120, use_best_model=True, **params)
model.fit(d.loc[fit_mask, FEATURES], d.loc[fit_mask, "target"],
          eval_set=(d.loc[val_mask, FEATURES], d.loc[val_mask, "target"]))

pred = model.predict(d.loc[te_mask, FEATURES]).astype(int).ravel()
y_te = d.loc[te_mask, "target"]
print(f"\n[holdout {TEST_YEAR}] macro F1={f1_score(y_te, pred, average='macro'):.4f} "
      f"acc={accuracy_score(y_te, pred):.4f}  (random baseline acc=0.333)")
print(classification_report(y_te, pred, target_names=["Under", "Neutral", "Outperform"]))

imp = pd.Series(model.get_feature_importance(), index=FEATURES).sort_values(ascending=False)
print("\n=== feature importance (raw OHLCV) ===")
print(imp.round(3).to_string())
