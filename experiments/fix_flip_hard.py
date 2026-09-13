"""fix_flip_hard — attack the direction coin-flip problem directly.

Honest diagnosis from the last run: when the ensemble commits to a direction
(predicts under or over, not neutral) on a fund that is truly extreme, it is
right 53.8% of the time -- barely better than a coin flip. Selecting only
the top-K%/day most-confident calls improves that number, but that dodges
the question instead of answering it.

This tries the one lever not yet tested: train the direction model on a much
MORE separated definition of "extreme" (top 5% / bottom 5% instead of
15/15), on the theory that the 15% cutoff includes a lot of barely-extreme,
ambiguous funds that dilute the direction signal. Sweeps the training
threshold (15/15, 10/10, 7/7, 5/5, 3/3) and reports, for EVERY threshold,
the UNFILTERED accuracy: of true extreme funds where the model commits to a
direction, how often is it right. No confidence gating, no top-K selection
-- this is the number that actually answers "is the coin flip fixed."

2-week horizon + tuned hyperparameters (both already shown to help), 1y
window, 2026 test only.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from exp_harness import load_universe, gpu_lock, EXP, SWEEP_FAMILIES

HORIZON, SKIP = 10, 0
EMBARGO = HORIZON + SKIP
TEST_YEAR = 2026
THRESHOLDS = [0.15, 0.10, 0.07, 0.05, 0.03]

TUNED = dict(max_depth=6, learning_rate=0.03, min_child_weight=8,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)

def stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s

def loo_peer_mean(x, cat):
    s = x.T.groupby(cat).transform("sum").T
    n = x.notna().T.groupby(cat).transform("sum").T
    return (s - x.fillna(0)) / (n - x.notna().astype(int)).replace(0, np.nan)

def build_label_h(close, cat, horizon, skip):
    fwd = close.shift(-horizon) / close.shift(-skip) - 1.0
    rel = fwd - loo_peer_mean(fwd, cat)
    fwd_l, rel_l = stack(fwd), stack(rel)
    target = (rel_l.dropna().groupby(level="date")
             .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                          labels=[0, 1, 2])).astype("int8"))
    return (pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1)
           .dropna(subset=["target"]))

frames = [pd.read_parquet(EXP / "features" / f) for f in SWEEP_FAMILIES
         if (EXP / "features" / f).exists()]
X = pd.concat(frames, axis=1)
X = X.loc[:, ~X.columns.duplicated(keep="first")].astype("float32")
X = X.mask(np.isinf(X))
features = list(X.columns)

fields, cat = load_universe()
close = fields["Close"]
lbl = build_label_h(close, cat, HORIZON, SKIP)
d = X.join(lbl, how="inner").dropna(subset=["target"])
d = d.dropna(subset=features, how="all")
d["target"] = d["target"].astype("int8")
date_index = d.index.get_level_values("date")
day_pct = d.groupby(level="date")["rel"].rank(pct=True)


def window_masks(years):
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
    test_end = pd.Timestamp(f"{TEST_YEAR}-12-31")
    cut = close.index[max(close.index.searchsorted(test_start) - EMBARGO, 0)]
    start = test_start - pd.DateOffset(years=years)
    tr = (date_index <= cut) & (date_index >= start)
    te = (date_index >= test_start) & (date_index <= test_end)
    tr_dates = np.sort(date_index[tr].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.90)]
    val_cut = close.index[max(close.index.searchsorted(val_start) - EMBARGO, 0)]
    fit = tr & (date_index <= val_cut)
    val = tr & (date_index >= val_start)
    return fit, val, te

fit_m, val_m, te_m = window_masks(1)
print(f"1y window: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}", flush=True)

from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, tree_method="hist", device="cuda",
           early_stopping_rounds=150, random_state=42, verbosity=0)

# fixed evaluation definition of "extreme" (15%) so every threshold's model
# is judged against the SAME test population -- otherwise tightening the
# threshold trivially "improves" accuracy just by testing on easier examples
d["y_top_eval"] = (day_pct >= 0.85).astype("int8")
d["y_ext_eval"] = ((day_pct >= 0.85) | (day_pct <= 0.15)).astype("int8")
tt_eval = te_m & (d["y_ext_eval"] == 1).values

rows = []
for thr in THRESHOLDS:
    y_top_tr = (day_pct >= 1.0 - thr).astype("int8")
    y_ext_tr = ((day_pct >= 1.0 - thr) | (day_pct <= thr)).astype("int8")
    ft = fit_m & (y_ext_tr == 1).values
    vt = val_m & (y_ext_tr == 1).values
    with gpu_lock():
        m = XGBClassifier(**BASE, **TUNED, objective="binary:logistic",
                          eval_metric="auc")
        m.fit(d.loc[ft, features], y_top_tr[ft],
             eval_set=[(d.loc[vt, features], y_top_tr[vt])], verbose=False)

    # judged on the SAME fixed 15% test population every time
    p = m.predict_proba(d.loc[tt_eval, features])[:, 1]
    y_true = d.loc[tt_eval, "y_top_eval"].to_numpy()
    auc = roc_auc_score(y_true, p)
    acc = accuracy_score(y_true, (p >= 0.5).astype(int))
    rows.append({"train_threshold_pct": thr, "n_train": int(ft.sum()),
                "eval_auc": round(float(auc), 4), "eval_accuracy": round(float(acc), 4),
                "best_iter": int(m.best_iteration)})
    print(f"train_thr={thr:.0%}  n_train={ft.sum():>7}  ->  "
         f"eval_auc={auc:.4f}  eval_acc={acc:.4f}  iters={m.best_iteration}", flush=True)

res = pd.DataFrame(rows)
print("\n===== training-threshold sweep (all judged on fixed 15% test set) =====")
print(res.to_string(index=False))
res.to_csv(EXP / "results" / "fix_flip_hard.csv", index=False)
print("\nsaved -> experiments/results/fix_flip_hard.csv")
