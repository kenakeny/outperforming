"""
train_basic_catboost.py — a *basic* CatBoost baseline built ONLY from the raw
yfinance download (OHLCV in data/raw/market_data.parquet + category metadata).

Purpose: establish what plain single-name technical features (returns, moving
averages, RSI/MACD/Bollinger, realized & intraday vol, volume) can predict about
next-5-day peer-relative outperformance, and read the feature importance so we
know which *directions* are worth engineering richer (peer-relative / cross-
sectional) features in next.

Deliberately NOT used here: features_v3.parquet and any of the peer-relative /
cross-sectional / leave-one-out engineered columns. Everything below is
computable from one ticker's own OHLCV, strictly trailing (<= t), so leak-free.

Label + holdout match train_5d.py exactly (5-trading-day horizon, no skip,
leave-one-out category-peer mean, per-day terciles, 5-day embargo, 2026 holdout)
so this baseline is directly comparable to the full-feature production model.

Run:  python train_basic_catboost.py
"""
import pathlib
import yaml
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ---------------- horizon / leakage config (same as train_5d.py) ----------------
HORIZON   = 5
SKIP      = 0
EMBARGO   = HORIZON + SKIP
MIN_GROUP = 3
TEST_YEAR = 2026
TASK_TYPE = "GPU"

ROOT = pathlib.Path(__file__).resolve().parent
cfg  = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RAW  = ROOT / cfg["paths"]["raw"]
PROC = ROOT / cfg["paths"]["processed"]

# ---------------- load the raw yfinance OHLCV ----------------
md = pd.read_parquet(RAW / "market_data.parquet")
md.index = pd.to_datetime(md.index); md.index.name = "date"
close  = md["Close"];  high = md["High"]; low = md["Low"]
openp  = md["Open"];   vol  = md["Volume"]
for f in (close, high, low, openp, vol):
    f.columns.name = "ticker"
trading_index = close.index
print("raw OHLCV:", close.shape, "|", trading_index.min().date(), "->", trading_index.max().date())

# ---------------- BASIC single-name features (all trailing, OHLCV only) ----------------
feat = {}
r1 = close.pct_change()

# --- momentum / returns ---
for n in (1, 5, 10, 20, 60):
    feat[f"ret_{n}d"] = close.pct_change(n)
feat["ret_20d_skip5"] = close.shift(5) / close.shift(25) - 1.0    # 20d momo, skip most-recent week
feat["ret_60d_skip5"] = close.shift(5) / close.shift(65) - 1.0

# --- realized & intraday volatility ---
feat["vol_20d"] = r1.rolling(20).std()
feat["vol_60d"] = r1.rolling(60).std()
tr = (high - low) / close                                         # daily true-ish range (intraday)
feat["hl_range_20d"] = tr.rolling(20).mean()
feat["gap_20d"] = (openp / close.shift(1) - 1.0).rolling(20).mean()   # avg overnight gap

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
feat["macd"]      = macd / close                # scale-free
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
feat["rel_volume_20d"]  = vol / vol.rolling(20).mean() - 1.0
feat["dollar_vol_log"]  = np.log1p((close * vol).rolling(20).mean())
dv = (close * vol)
feat["vol_zscore_20d"]  = (dv - dv.rolling(20).mean()) / dv.rolling(20).std()

FEATURES = list(feat.keys())
print(f"basic features ({len(FEATURES)}):", FEATURES)

# stack each wide frame -> long (date, ticker) and assemble the design matrix
def _stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s

X = pd.DataFrame({name: _stack(df) for name, df in feat.items()})

# ---------------- label: next-5d peer-relative tercile (identical to train_5d.py) ----------------
metadata = pd.read_parquet(RAW / "metadata.parquet")
cat = metadata["category"].reindex(close.columns)
counts = cat.value_counts()
keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
kept_cols = cat[cat.isin(keep)].index
px, cat = close[kept_cols], cat[kept_cols]

fwd = px.shift(-HORIZON) / px.shift(-SKIP) - 1.0                  # TARGET window t+1..t+5
grp_sum = fwd.T.groupby(cat).transform("sum").T
grp_n   = fwd.notna().T.groupby(cat).transform("sum").T
peer_mean = (grp_sum - fwd.fillna(0)) / (grp_n - fwd.notna().astype(int)).replace(0, np.nan)
rel = fwd - peer_mean                                             # leave-one-out peer-relative excess

fwd_l, rel_l = _stack(fwd), _stack(rel)
target = (rel_l.dropna().groupby(level="date")
          .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=[0, 1, 2])).astype("int8"))
lbl = pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1).dropna(subset=["target"])

# ---------------- assemble labeled dataset ----------------
d = X.join(lbl, how="inner").dropna(subset=["target"])
d = d.dropna(subset=FEATURES, how="all")
d["target"] = d["target"].astype("int8")
print("labeled dataset:", d.shape,
      "| class balance:", dict(d["target"].value_counts(normalize=True).round(2)))

date_index = d.index.get_level_values("date")
test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
cut = trading_index[max(trading_index.searchsorted(test_start) - EMBARGO, 0)]
tr_mask = date_index <= cut
te_mask = date_index >= test_start

tr_dates = np.sort(date_index[tr_mask].unique())
val_start = tr_dates[int(len(tr_dates) * 0.90)]
val_cut = trading_index[max(trading_index.searchsorted(val_start) - EMBARGO, 0)]
fit_mask = tr_mask & (date_index <= val_cut)
val_mask = tr_mask & (date_index >= val_start)

params = dict(
    iterations=2000, learning_rate=0.04, depth=7, l2_leaf_reg=3,
    loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
    random_seed=42, task_type=TASK_TYPE, verbose=200,
)

print("\ntraining basic-feature holdout model ...")
model = CatBoostClassifier(early_stopping_rounds=120, use_best_model=True, **params)
model.fit(d.loc[fit_mask, FEATURES], d.loc[fit_mask, "target"],
          eval_set=(d.loc[val_mask, FEATURES], d.loc[val_mask, "target"]))

pred = model.predict(d.loc[te_mask, FEATURES]).astype(int).ravel()
y_te = d.loc[te_mask, "target"]
print(f"\n[holdout {TEST_YEAR}] macro F1={f1_score(y_te, pred, average='macro'):.4f} "
      f"acc={accuracy_score(y_te, pred):.4f}  (random baseline acc=0.333)")
print(classification_report(y_te, pred, target_names=["Under", "Neutral", "Outperform"]))

proba_te = model.predict_proba(d.loc[te_mask, FEATURES])[:, 2]
ic_df = pd.DataFrame({"score": proba_te, "rel": d.loc[te_mask, "rel"]}, index=y_te.index)
ic = ic_df.groupby(level="date").apply(lambda g: g["score"].corr(g["rel"], method="spearman"))
print(f"holdout rank IC (mean daily Spearman of P(outperform) vs peer-rel return): {ic.mean():.4f}")

# ---------------- FEATURE IMPORTANCE ----------------
imp = (pd.Series(model.get_feature_importance(), index=FEATURES)
       .sort_values(ascending=False))
print("\n=== CatBoost feature importance (PredictionValuesChange) ===")
print(imp.round(3).to_string())
imp.rename("importance").to_csv(PROC / "basic_feature_importance.csv", header=True)
print("\nsaved ->", PROC / "basic_feature_importance.csv")
