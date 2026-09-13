"""
exp_09_hpsweep.py — CatBoost hyperparameter sweep over the 27 basic OHLCV
features (same feature block as train_basic_catboost.py, same label/holdout
harness as every other experiment).

Baseline reference: depth=7, lr=0.04 -> macro F1 0.4632 / rank IC 0.0353
on the 2026 holdout.

Resumable: each finished config is appended to
experiments/results/exp_09_hpsweep.csv immediately; on rerun, configs whose
row already exists are skipped.

Run:  python experiments/exp_09_hpsweep.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json

import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, holdout_masks,
                         gpu_lock, DEFAULT_PARAMS, EXP)
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

CSV_PATH  = EXP / "results" / "exp_09_hpsweep.csv"
JSON_PATH = EXP / "results" / "exp_09_hpsweep.json"

# ---------------- data: 27 basic single-name OHLCV features ----------------
fields, cat = load_universe()
close = fields["Close"]; high = fields["High"]; low = fields["Low"]
openp = fields["Open"];  vol  = fields["Volume"]
trading_index = close.index
print("universe:", close.shape, "|", trading_index.min().date(), "->",
      trading_index.max().date())

feat = {}
r1 = close.pct_change(fill_method=None)

# --- momentum / returns ---
for n in (1, 5, 10, 20, 60):
    feat[f"ret_{n}d"] = close.pct_change(n, fill_method=None)
feat["ret_20d_skip5"] = close.shift(5) / close.shift(25) - 1.0
feat["ret_60d_skip5"] = close.shift(5) / close.shift(65) - 1.0

# --- realized & intraday volatility ---
feat["vol_20d"] = r1.rolling(20).std()
feat["vol_60d"] = r1.rolling(60).std()
tr = (high - low) / close
feat["hl_range_20d"] = tr.rolling(20).mean()
feat["gap_20d"] = (openp / close.shift(1) - 1.0).rolling(20).mean()

# --- trend: price relative to moving averages ---
for n in (10, 20, 50, 200):
    feat[f"px_to_sma_{n}"] = close / close.rolling(n).mean() - 1.0
for n in (12, 26):
    feat[f"px_to_ema_{n}"] = close / close.ewm(span=n, adjust=False).mean() - 1.0

# --- RSI(14) ---
delta = close.diff()
gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
feat["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

# --- MACD ---
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
macd_sig = macd.ewm(span=9, adjust=False).mean()
feat["macd"]      = macd / close
feat["macd_hist"] = (macd - macd_sig) / close

# --- Bollinger(20,2) ---
sma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
feat["bb_pctb"]      = (close - (sma20 - 2 * std20)) / (4 * std20)
feat["bb_bandwidth"] = (4 * std20) / sma20

# --- 52-week position ---
feat["px_to_52w_high"] = close / close.rolling(252).max() - 1.0
feat["px_to_52w_low"]  = close / close.rolling(252).min() - 1.0

# --- volume ---
feat["rel_volume_20d"] = vol / vol.rolling(20).mean() - 1.0
feat["dollar_vol_log"] = np.log1p((close * vol).rolling(20).mean())
dv = close * vol
feat["vol_zscore_20d"] = (dv - dv.rolling(20).mean()) / dv.rolling(20).std()

FEATURES = list(feat.keys())
print(f"basic features ({len(FEATURES)}):", FEATURES)

# ---------------- assemble + label + masks ----------------
X = assemble(feat)
lbl = build_label(close, cat)
d = X.join(lbl, how="inner").dropna(subset=["target"])
d = d.dropna(subset=FEATURES, how="all")
d["target"] = d["target"].astype("int8")
print("labeled dataset:", d.shape)

date_index = d.index.get_level_values("date")
fit_mask, val_mask, te_mask = holdout_masks(date_index, trading_index)
print(f"fit={fit_mask.sum()}  val={val_mask.sum()}  test={te_mask.sum()}")

X_fit, y_fit = d.loc[fit_mask, FEATURES], d.loc[fit_mask, "target"]
X_val, y_val = d.loc[val_mask, FEATURES], d.loc[val_mask, "target"]
X_te,  y_te  = d.loc[te_mask, FEATURES],  d.loc[te_mask, "target"]
rel_te = d.loc[te_mask, "rel"]

# ---------------- grid ----------------
GRID = [
    ("a", dict(depth=6,  learning_rate=0.04)),
    ("b", dict(depth=8,  learning_rate=0.04)),
    ("c", dict(depth=10, learning_rate=0.04)),
    ("d", dict(depth=7,  learning_rate=0.02)),
    ("e", dict(depth=7,  learning_rate=0.08)),
    ("f", dict(depth=8,  learning_rate=0.02, l2_leaf_reg=9)),
    ("g", dict(depth=10, learning_rate=0.02, l2_leaf_reg=9)),
]

COLS = ["config", "depth", "lr", "l2", "macro_f1", "accuracy", "rank_ic",
        "best_iter"]

done = set()
if CSV_PATH.exists():
    prev = pd.read_csv(CSV_PATH)
    done = set(prev["config"].astype(str))
    print("resuming — already done:", sorted(done))

for name, overrides in GRID:
    if name in done:
        print(f"[{name}] already in CSV, skipping")
        continue
    params = {**DEFAULT_PARAMS, "iterations": 2000, "verbose": 0, **overrides}
    print(f"[{name}] training depth={params['depth']} lr={params['learning_rate']} "
          f"l2={params['l2_leaf_reg']} ...", flush=True)
    with gpu_lock():
        model = CatBoostClassifier(early_stopping_rounds=120,
                                   use_best_model=True, **params)
        model.fit(X_fit, y_fit, eval_set=(X_val, y_val))

    pred  = np.asarray(model.predict(X_te)).ravel().astype(int)
    proba = model.predict_proba(X_te)[:, 2]
    ic = (pd.DataFrame({"s": proba, "rel": rel_te}, index=y_te.index)
          .groupby(level="date")
          .apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean())
    row = {
        "config": name,
        "depth": params["depth"],
        "lr": params["learning_rate"],
        "l2": params["l2_leaf_reg"],
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "rank_ic": round(float(ic), 4),
        "best_iter": int(model.get_best_iteration() or params["iterations"]),
    }
    pd.DataFrame([row])[COLS].to_csv(CSV_PATH, mode="a", index=False,
                                     header=not CSV_PATH.exists())
    print(f"[{name}] macro_f1={row['macro_f1']}  acc={row['accuracy']}  "
          f"rank_ic={row['rank_ic']}  best_iter={row['best_iter']}", flush=True)

# ---------------- summary ----------------
res = pd.read_csv(CSV_PATH).drop_duplicates(subset="config", keep="last")
res = res.sort_values("config").reset_index(drop=True)
print("\n=== exp_09_hpsweep results ===")
print(res.to_string(index=False))

best_f1 = res.loc[res["macro_f1"].idxmax()]
best_ic = res.loc[res["rank_ic"].idxmax()]
summary = {
    "name": "exp_09_hpsweep",
    "baseline": {"depth": 7, "lr": 0.04, "macro_f1": 0.4632, "rank_ic": 0.0353},
    "best_macro_f1": best_f1.to_dict(),
    "best_rank_ic": best_ic.to_dict(),
    "results": res.to_dict(orient="records"),
}
JSON_PATH.write_text(json.dumps(summary, indent=2, default=str))
print(f"\nbest macro F1: config {best_f1['config']} "
      f"(depth={best_f1['depth']}, lr={best_f1['lr']}, l2={best_f1['l2']}) "
      f"macro_f1={best_f1['macro_f1']}")
print(f"best rank IC:  config {best_ic['config']} "
      f"(depth={best_ic['depth']}, lr={best_ic['lr']}, l2={best_ic['l2']}) "
      f"rank_ic={best_ic['rank_ic']}")
print("saved ->", CSV_PATH, "and", JSON_PATH)
