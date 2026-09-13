"""
train_5d.py — retrain the peer-relative outperformance model for a 5-TRADING-DAY horizon.

The features do NOT change: they're all trailing (computed from data <= t), so they're
horizon-agnostic and stay leak-free at any horizon. The ONLY things that change for a
"next 5 days" target are the label window and the leakage controls:

    * forward-return window : next 5 days, NO skip        (was: skip 5, hold 20)
    * train/test embargo    : 5 days  (= label horizon)   (was: 25)

Pipeline:  build 5d peer-relative tercile label -> join to features_v3 -> honest holdout
eval (+ rank IC) -> refit on ALL labeled data -> save model_5d.cbm -> print the live
"next 5 days" picks for the most recent date.

Run:  python train_5d.py
"""
import pathlib
import yaml
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ---------------- horizon / leakage config (the ONLY things that change vs the 20d model) ----------------
HORIZON   = 5     # predict the next 5 trading days
SKIP      = 0     # no skip — we want the immediate next-5-day window
EMBARGO   = HORIZON + SKIP        # purge gap between train and test MUST equal the label horizon
MIN_GROUP = 3
TEST_YEAR = 2026  # held out for honest evaluation
TASK_TYPE = "GPU"  # -> "CPU" if no GPU

ROOT = pathlib.Path(__file__).resolve().parent
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
cfg  = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RAW  = ROOT / cfg["paths"]["raw"]
PROC = ROOT / cfg["paths"]["processed"]

# ---------------- features (unchanged, all trailing -> reusable at any horizon) ----------------
features_long = pd.read_parquet(PROC / "features_v3.parquet")
features_long.index = features_long.index.set_levels(
    pd.to_datetime(features_long.index.levels[0]), level="date")
print("features:", features_long.shape)

# ---------------- build the 5-day peer-relative label ----------------
metadata = pd.read_parquet(RAW / "metadata.parquet")
prices   = pd.read_parquet(RAW / "prices.parquet")
prices.index.name = "date"; prices.columns.name = "ticker"

cat = metadata["category"].reindex(prices.columns)
counts = cat.value_counts()
keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
cols = cat[cat.isin(keep)].index
px, cat = prices[cols], cat[cols]

# forward return over the NEXT 5 days (no skip). Uses prices t+1..t+5 — that's the TARGET,
# not a feature; every feature is still strictly <= t, so no leakage.
fwd = px.shift(-HORIZON) / px.shift(-SKIP) - 1.0

# leave-one-out category peer mean -> peer-relative excess
grp_sum = fwd.T.groupby(cat).transform("sum").T
grp_n   = fwd.notna().T.groupby(cat).transform("sum").T
peer_mean = (grp_sum - fwd.fillna(0)) / (grp_n - fwd.notna().astype(int)).replace(0, np.nan)
rel = fwd - peer_mean

def _stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s

fwd_l, rel_l = _stack(fwd), _stack(rel)
# per-day terciles of peer-relative return -> balanced 0/1/2 every day (no regime artifact)
target = (rel_l.dropna().groupby(level="date")
          .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=[0, 1, 2])).astype("int8"))

lbl = pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1).dropna(subset=["target"])

# ---------------- join label to features ----------------
dataset = features_long.copy()
dataset["target"]  = lbl["target"].reindex(dataset.index)
dataset["fwd_ret"] = lbl["fwd_ret"].reindex(dataset.index)
dataset["rel"]     = lbl["rel"].reindex(dataset.index)
dataset = dataset.dropna(subset=["target"])
dataset["target"] = dataset["target"].astype("int8")
print("labeled dataset:", dataset.shape,
      "| class balance:", dict(dataset["target"].value_counts(normalize=True).round(2)))

NON_FEATURES = ["target", "fwd_ret", "rel"]
features = [c for c in dataset.columns if c not in NON_FEATURES]

# ---------------- honest holdout: train < TEST_YEAR (embargoed) -> test on TEST_YEAR ----------------
date_index = dataset.index.get_level_values("date")
test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
cut = test_start - pd.offsets.BDay(EMBARGO)
tr_mask = date_index <= cut
te_mask = date_index >= test_start

# validation tail inside train for early stopping (also embargoed)
tr_dates = np.sort(date_index[tr_mask].unique())
val_start = tr_dates[int(len(tr_dates) * 0.90)]
val_cut = val_start - pd.offsets.BDay(EMBARGO)
fit_mask = tr_mask & (date_index <= val_cut)
val_mask = tr_mask & (date_index >= val_start)

params = dict(
    iterations=3000, learning_rate=0.04, depth=7, l2_leaf_reg=3,
    loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
    random_seed=42, task_type=TASK_TYPE, verbose=300,
)

print("\ntraining holdout model ...")
eval_model = CatBoostClassifier(early_stopping_rounds=150, use_best_model=True, **params)
eval_model.fit(dataset.loc[fit_mask, features], dataset.loc[fit_mask, "target"],
               eval_set=(dataset.loc[val_mask, features], dataset.loc[val_mask, "target"]))

pred = eval_model.predict(dataset.loc[te_mask, features]).astype(int).ravel()
y_te = dataset.loc[te_mask, "target"]
print(f"\n[holdout {TEST_YEAR}] macro F1={f1_score(y_te, pred, average='macro'):.4f} "
      f"acc={accuracy_score(y_te, pred):.4f}")
print(classification_report(y_te, pred, target_names=["Under", "Neutral", "Outperform"]))

# rank IC — the cross-sectional metric that actually matters (daily Spearman of score vs rel)
proba_te = eval_model.predict_proba(dataset.loc[te_mask, features])[:, 2]
ic_df = pd.DataFrame({"score": proba_te, "rel": dataset.loc[te_mask, "rel"]}, index=y_te.index)
ic = ic_df.groupby(level="date").apply(lambda d: d["score"].corr(d["rel"], method="spearman"))
print(f"holdout rank IC (mean daily Spearman): {ic.mean():.4f}")

# save a leakage-free backtest frame (HOLDOUT model — never saw the test year)
_price_l = prices.stack(); _price_l.index.names = ["date", "ticker"]
pd.DataFrame({
    "score":   proba_te,
    "fwd_ret": dataset.loc[te_mask, "fwd_ret"].values,
    "price":   _price_l.reindex(y_te.index).values,
}, index=y_te.index).to_parquet(PROC / "backtest_5d.parquet")
print("saved backtest frame ->", PROC / "backtest_5d.parquet")

# ---------------- refit on ALL labeled data for deployment ----------------
best_iters = eval_model.get_best_iteration() or params["iterations"]
print(f"\nrefitting final model on all labeled data ({best_iters} iters) ...")
final_model = CatBoostClassifier(**{**params, "iterations": best_iters})
final_model.fit(dataset[features], dataset["target"])
final_model.save_model(str(PROC / "model_5d.cbm"))
print("saved ->", PROC / "model_5d.cbm")

# ---------------- live signal: score the most recent date (its 5d return isn't known yet) ----------------
live_date = pd.Timestamp(np.sort(features_long.index.get_level_values("date").unique())[-1])
live = features_long.loc[features_long.index.get_level_values("date") == live_date, features]
price_l = prices.stack(); price_l.index.names = ["date", "ticker"]
live = live[price_l.reindex(live.index).notna()]   # only funds actually trading that day

p_out = final_model.predict_proba(live)[:, 2]
picks = pd.Series(p_out, index=live.index.get_level_values("ticker")).sort_values(ascending=False)
print(f"\n=== next-{HORIZON}-day outperformance signal - as of {live_date.date()} ===")
print("top 20 ETFs by P(outperform):")
print(picks.head(20).round(3).to_string())
picks.to_csv(PROC / "signal_5d.csv", header=["p_outperform"])
print("\nfull ranked signal saved -> signal_5d.csv")
