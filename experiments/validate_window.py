"""validate_window — 2026 holdout, varying the training-window length.

The walk-forward showed every model going dark in 2022 (regime break), which
raises the question: is old data helping or hurting? Here the test stays
fixed (2026, embargoed as always) and the train window shrinks: only the
most recent 6m / 1y / 2y / 3y / 4y / 5y of data before the cutoff is used
(rolling window instead of the expanding window used everywhere else).
Last 10% of each window's dates = early-stopping val, embargoed like
holdout_masks. Models: multiclass XGBoost (reference) and the validated
2-stage ensemble (extreme gate x tails direction).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, gpu_lock, EMBARGO, EXP

TEST_YEAR, TOP_PCT = 2026, 0.15
WINDOWS = [("6m", pd.DateOffset(months=6)), ("1y", pd.DateOffset(years=1)),
           ("2y", pd.DateOffset(years=2)), ("3y", pd.DateOffset(years=3)),
           ("4y", pd.DateOffset(years=4)), ("5y", pd.DateOffset(years=5))]
OUT = EXP / "results" / "validate_window.csv"

d, features, _, _, _, trading_index = load_sweep_dataset()
date_index = d.index.get_level_values("date")

day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y_top"] = (day_pct >= 1.0 - TOP_PCT).astype("int8")
d["y_ext"] = ((day_pct >= 1.0 - TOP_PCT) | (day_pct <= TOP_PCT)).astype("int8")

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score

BASE = dict(n_estimators=3000, learning_rate=0.03, max_depth=8,
            min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, tree_method="hist", device="cuda",
            early_stopping_rounds=150, random_state=42, verbosity=0)


def window_masks(offset):
    """holdout_masks logic with an additional rolling-window start bound."""
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
    test_end = pd.Timestamp(f"{TEST_YEAR}-12-31")
    cut = trading_index[max(trading_index.searchsorted(test_start) - EMBARGO, 0)]
    start = test_start - offset
    tr = (date_index <= cut) & (date_index >= start)
    te = (date_index >= test_start) & (date_index <= test_end)
    tr_dates = np.sort(date_index[tr].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.90)]
    val_cut = trading_index[max(trading_index.searchsorted(val_start) - EMBARGO, 0)]
    fit = tr & (date_index <= val_cut)
    val = tr & (date_index >= val_start)
    return fit, val, te


def fit_xgb(mask_fit, mask_val, ycol, **kw):
    with gpu_lock():
        m = XGBClassifier(**{**BASE, **kw})
        m.fit(d.loc[mask_fit, features], d.loc[mask_fit, ycol],
              eval_set=[(d.loc[mask_val, features], d.loc[mask_val, ycol])],
              verbose=False)
    return m


def score_stats(s, te_mask):
    te = pd.DataFrame({"s": s, "rel": d.loc[te_mask, "rel"],
                       "y_top": d.loc[te_mask, "y_top"]},
                      index=d.index[te_mask])
    g = te.groupby(level="date")
    ic = g.apply(lambda x: x["s"].corr(x["rel"], method="spearman")).mean()
    picks = g.apply(lambda x: x.nlargest(max(int(len(x) * TOP_PCT), 1), "s")
                    [["y_top", "rel"]].mean())
    return (round(float(ic), 4), round(float(picks["y_top"].mean()), 4),
            round(float(picks["rel"].mean()), 5))


rows = []
for wname, offset in WINDOWS:
    fit_m, val_m, te_m = window_masks(offset)
    print(f"\n===== window {wname}: fit={fit_m.sum()} val={val_m.sum()} "
          f"test={te_m.sum()}", flush=True)

    mc = fit_xgb(fit_m, val_m, "target",
                 objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    proba = mc.predict_proba(d.loc[te_m, features])
    ic, p15, rel15 = score_stats(proba[:, 2], te_m)
    mf1 = round(float(f1_score(d.loc[te_m, "target"], proba.argmax(1),
                               average="macro")), 4)
    rows.append({"window": wname, "model": "mc_xgb", "n_fit": int(fit_m.sum()),
                 "rank_ic": ic, "prec_top15": p15, "rel_top15": rel15,
                 "macro_f1": mf1})
    print(f"  mc_xgb   ic={ic}  prec15={p15}  rel15={rel15}  f1={mf1}", flush=True)

    ft, vt = fit_m & (d["y_ext"] == 1).values, val_m & (d["y_ext"] == 1).values
    tl = fit_xgb(ft, vt, "y_top", objective="binary:logistic", eval_metric="auc")
    s_dir = tl.predict_proba(d.loc[te_m, features])[:, 1]
    tt = te_m & (d["y_ext"] == 1).values
    auc_t = round(float(roc_auc_score(
        d.loc[tt, "y_top"], tl.predict_proba(d.loc[tt, features])[:, 1])), 4)

    y_ext_fit = d.loc[fit_m, "y_ext"]
    spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
    ex = fit_xgb(fit_m, val_m, "y_ext", objective="binary:logistic",
                 scale_pos_weight=spw, eval_metric="aucpr")
    p_ext = ex.predict_proba(d.loc[te_m, features])[:, 1]
    s_ens = p_ext * (2.0 * s_dir - 1.0)
    ic, p15, rel15 = score_stats(s_ens, te_m)
    rows.append({"window": wname, "model": "ensemble", "n_fit": int(fit_m.sum()),
                 "rank_ic": ic, "prec_top15": p15, "rel_top15": rel15,
                 "tail_auc": auc_t})
    print(f"  ensemble ic={ic}  prec15={p15}  rel15={rel15}  tail_auc={auc_t}",
          flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)

res = pd.DataFrame(rows)
print("\n===== results by training window =====")
print(res.to_string(index=False))
print("\nsaved ->", OUT)
