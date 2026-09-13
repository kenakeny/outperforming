"""validate_oos — expanding walk-forward validation (2021-2026) + 2-stage ensemble.

Everything so far was judged on the single 2026 holdout. This refits each
contender per test year with the same embargoed expanding-window scheme
(holdout_masks), so we see whether the 2026 story holds out-of-sample:

  mc     — multiclass XGBoost (best sweep model), score = P(outperform)
  tails  — binary top-15% vs bottom-15% (direction), scored on all funds
  ens    — 2-stage ensemble: stage 1 gates extremeness (both tails vs the
           middle 70%, scale_pos_weight-balanced), stage 2 is the tails
           direction model; score = P(extreme) * (2*P(top|extreme) - 1)

Per (year, model): rank IC (daily spearman of score vs realized peer-rel
excess), precision of daily top-15% picks (base 0.15), and the money metric —
the mean realized 5d peer-relative excess of those picks. Macro F1 for mc,
tail AUC for the direction stage. Appends to results CSV after each year so
a crash loses nothing.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, holdout_masks, gpu_lock, EXP

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
TOP_PCT = 0.15
OUT = EXP / "results" / "validate_oos.csv"

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
for year in YEARS:
    fit_m, val_m, te_m = holdout_masks(date_index, trading_index, year)
    if te_m.sum() == 0:
        continue
    print(f"\n===== test year {year}: fit={fit_m.sum()} val={val_m.sum()} "
          f"test={te_m.sum()}", flush=True)

    # --- multiclass (sweep winner) ------------------------------------
    mc = fit_xgb(fit_m, val_m, "target",
                 objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    proba = mc.predict_proba(d.loc[te_m, features])
    ic, p15, rel15 = score_stats(proba[:, 2], te_m)
    mf1 = round(float(f1_score(d.loc[te_m, "target"], proba.argmax(1),
                               average="macro")), 4)
    rows.append({"year": year, "model": "mc_xgb", "rank_ic": ic,
                 "prec_top15": p15, "rel_top15": rel15, "macro_f1": mf1})
    print(f"  mc_xgb   ic={ic}  prec15={p15}  rel15={rel15}  f1={mf1}", flush=True)

    # --- tails direction ----------------------------------------------
    ft, vt = fit_m & (d["y_ext"] == 1).values, val_m & (d["y_ext"] == 1).values
    tl = fit_xgb(ft, vt, "y_top", objective="binary:logistic", eval_metric="auc")
    s_dir = tl.predict_proba(d.loc[te_m, features])[:, 1]
    tt = te_m & (d["y_ext"] == 1).values
    auc_t = round(float(roc_auc_score(
        d.loc[tt, "y_top"], tl.predict_proba(d.loc[tt, features])[:, 1])), 4)
    ic, p15, rel15 = score_stats(s_dir, te_m)
    rows.append({"year": year, "model": "tails", "rank_ic": ic,
                 "prec_top15": p15, "rel_top15": rel15, "tail_auc": auc_t})
    print(f"  tails    ic={ic}  prec15={p15}  rel15={rel15}  auc={auc_t}", flush=True)

    # --- 2-stage ensemble ---------------------------------------------
    y_ext_fit = d.loc[fit_m, "y_ext"]
    spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
    ex = fit_xgb(fit_m, val_m, "y_ext", objective="binary:logistic",
                 scale_pos_weight=spw, eval_metric="aucpr")
    p_ext = ex.predict_proba(d.loc[te_m, features])[:, 1]
    s_ens = p_ext * (2.0 * s_dir - 1.0)
    ic, p15, rel15 = score_stats(s_ens, te_m)
    rows.append({"year": year, "model": "ensemble", "rank_ic": ic,
                 "prec_top15": p15, "rel_top15": rel15})
    print(f"  ensemble ic={ic}  prec15={p15}  rel15={rel15}", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)

res = pd.DataFrame(rows)
print("\n===== per-year results =====")
print(res.to_string(index=False))
print("\n===== means across years =====")
print(res.groupby("model")[["rank_ic", "prec_top15", "rel_top15"]]
      .mean().round(4).to_string())
res.to_csv(OUT, index=False)
print("\nsaved ->", OUT)
