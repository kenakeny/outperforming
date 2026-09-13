"""model_ensemble_oos — multi-year out-of-sample validation + 2-stage ensemble.

Walk-forward over test years 2022..2026 (expanding train window, 5-day
embargo via exp_harness.holdout_masks — same purge discipline as everything
else). For every year, four models are trained on identical splits:

  mc          multiclass XGBoost (the sweep winner); ranking score = P2 - P0
  top_rest    binary top-15% vs rest, class-weighted   (known vol-detector)
  tails       binary top-15% vs bottom-15% (direction specialist)
  ensemble    stage 1: extreme (either tail, 30%) vs neutral, class-weighted
              stage 2: the `tails` direction model
              score = P(extreme) * (2*P(top|tail) - 1)

Metrics per (year, model), all on that year's full daily cross-section:
  rank_ic      mean daily spearman(score, realized peer-relative excess)
  prec_top15   of the model's daily top-15% picks, fraction that landed in
               the true top 15% (base rate 0.15)
Plus dir_acc: accuracy of top-vs-bottom calls on true tail rows only.

Writes experiments/results/oos_validation.csv and prints pivot tables.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, time
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, holdout_masks, gpu_lock, EXP
from xgboost import XGBClassifier

TOP_PCT = 0.15
YEARS = [2022, 2023, 2024, 2025, 2026]

XGB = dict(n_estimators=1500, learning_rate=0.03, max_depth=8,
           min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
           reg_lambda=5.0, tree_method="hist", device="cuda",
           early_stopping_rounds=100, random_state=42, verbosity=0)

d, features, _, _, _, trading_index = load_sweep_dataset()
date_index = d.index.get_level_values("date")
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y_top"] = (day_pct >= 1 - TOP_PCT).astype("int8")
d["y_ex"] = ((day_pct >= 1 - TOP_PCT) | (day_pct <= TOP_PCT)).astype("int8")
print("dataset:", d.shape, flush=True)


def fit_bin(fit_m, val_m, ycol, weighted):
    y = d.loc[fit_m, ycol]
    spw = float((y == 0).sum() / (y == 1).sum()) if weighted else 1.0
    m = XGBClassifier(objective="binary:logistic", eval_metric="auc",
                      scale_pos_weight=spw, **XGB)
    m.fit(d.loc[fit_m, features], y,
          eval_set=[(d.loc[val_m, features], d.loc[val_m, ycol])], verbose=False)
    return m


def daily_metrics(score, te_m):
    fr = pd.DataFrame({"s": score, "rel": d.loc[te_m, "rel"],
                       "y": d.loc[te_m, "y_top"]}, index=d.index[te_m])
    g = fr.groupby(level="date")
    ic = g.apply(lambda x: x["s"].corr(x["rel"], method="spearman")).mean()
    prec = g.apply(lambda x: x.nlargest(max(int(len(x) * TOP_PCT), 1), "s")["y"].mean()).mean()
    return round(float(ic), 4), round(float(prec), 4)


rows = []
with gpu_lock():
    for year in YEARS:
        t0 = time.time()
        fit_m, val_m, te_m = holdout_masks(date_index, trading_index, year)
        tail_m = d["y_ex"].to_numpy().astype(bool)
        te_tails = te_m & tail_m
        Xte = d.loc[te_m, features]
        y_dir_true = d.loc[te_tails, "y_top"].to_numpy()

        # --- multiclass reference
        m = XGBClassifier(objective="multi:softprob", num_class=3,
                          eval_metric="mlogloss", **XGB)
        m.fit(d.loc[fit_m, features], d.loc[fit_m, "target"],
              eval_set=[(d.loc[val_m, features], d.loc[val_m, "target"])],
              verbose=False)
        p = m.predict_proba(Xte)
        ic, prec = daily_metrics(p[:, 2] - p[:, 0], te_m)
        pt = m.predict_proba(d.loc[te_tails, features])
        dir_acc = float(((pt[:, 2] > pt[:, 0]).astype(int) == y_dir_true).mean())
        rows.append(dict(year=year, model="mc", rank_ic=ic, prec_top15=prec,
                         dir_acc=round(dir_acc, 4)))

        # --- binary top-15 vs rest (weighted)
        m = fit_bin(fit_m, val_m, "y_top", weighted=True)
        ic, prec = daily_metrics(m.predict_proba(Xte)[:, 1], te_m)
        pt = m.predict_proba(d.loc[te_tails, features])[:, 1]
        dir_acc = float(((pt >= np.median(pt)).astype(int) == y_dir_true).mean())
        rows.append(dict(year=year, model="top_rest", rank_ic=ic, prec_top15=prec,
                         dir_acc=round(dir_acc, 4)))

        # --- binary tails (direction specialist)
        mt = fit_bin(fit_m & tail_m, val_m & tail_m, "y_top", weighted=False)
        s_dir_all = 2 * mt.predict_proba(Xte)[:, 1] - 1
        ic, prec = daily_metrics(s_dir_all, te_m)
        pt = mt.predict_proba(d.loc[te_tails, features])[:, 1]
        dir_acc = float(((pt >= 0.5).astype(int) == y_dir_true).mean())
        rows.append(dict(year=year, model="tails", rank_ic=ic, prec_top15=prec,
                         dir_acc=round(dir_acc, 4)))

        # --- ensemble: P(extreme) * direction
        me = fit_bin(fit_m, val_m, "y_ex", weighted=True)
        s_ex = me.predict_proba(Xte)[:, 1]
        ic, prec = daily_metrics(s_ex * s_dir_all, te_m)
        rows.append(dict(year=year, model="ensemble", rank_ic=ic, prec_top15=prec,
                         dir_acc=None))

        print(f"[{year}] done in {(time.time()-t0)/60:.1f}m  "
              f"n_fit={fit_m.sum()}  n_test={te_m.sum()}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(EXP / "results" / "oos_validation.csv", index=False)

for metric in ["rank_ic", "prec_top15", "dir_acc"]:
    piv = res.pivot(index="model", columns="year", values=metric)
    piv["mean"] = piv.mean(axis=1)
    print(f"\n=== {metric} ===")
    print(piv.round(4).to_string())
