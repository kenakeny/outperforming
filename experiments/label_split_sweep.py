"""label_split_sweep — redefine the classes: does a rarer, more extreme
under/over definition (80/10/10, 90/5/5) classify better AND make money?

For each split t in {15%, 10%, 5%} (i.e. 70/15/15, 80/10/10, 90/5/5):
  - label: over = top t% of daily peer-relative 10d excess, under = bottom
    t%, neutral = the rest
  - model: the validated two-stage XGB (extreme(2t)-vs-rest gate, then
    top-vs-bottom direction inside the tails), tuned stage2 hyperparams,
    1y rolling window
  - prediction at matched rates: top t% of the blend-style score per day
    -> "over", bottom t% -> "under" (predicted class rates == label rates,
    so per-split F1 numbers are comparable and honest)
  - classification: macro F1, per-class F1, precision of over/under calls
  - MONEY: long predicted overs, short predicted unders, 10-trading-day
    hold, overlapping daily formation. Approximations, stated plainly:
    each formation day contributes its calls' mean signed rel (the realized
    10d peer-relative excess); cost = 20 bps per call round trip (10 bps
    per side); annualization uses sqrt(252/10) on the formation-day series
    (non-overlapping approximation).

2026 test year. All models GPU-locked. Results -> results/label_split_sweep.csv
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score
from exp_harness import load_sweep_dataset, gpu_lock, EXP

HORIZON = 10
SPLITS = [0.15, 0.10, 0.05]
COST_RT = 0.002          # 20 bps round trip per call

d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                        window_years=1)
print(f"fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}", flush=True)
day_pct = d.groupby(level="date")["rel"].rank(pct=True)

from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, tree_method="hist", device="cuda",
            early_stopping_rounds=150, random_state=42, verbosity=0)
TUNED = dict(max_depth=6, learning_rate=0.03, min_child_weight=8,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)

rows = []
for t in SPLITS:
    y3 = np.select([day_pct <= t, day_pct >= 1 - t], [0, 2], default=1)
    y3 = pd.Series(y3, index=d.index)
    y_top = (day_pct >= 1 - t).astype("int8")
    y_ext = ((day_pct >= 1 - t) | (day_pct <= t)).astype("int8")

    ft, vt = fit_m & (y_ext == 1).values, val_m & (y_ext == 1).values
    with gpu_lock():
        stage2 = XGBClassifier(**BASE, **TUNED, objective="binary:logistic",
                               eval_metric="auc")
        stage2.fit(d.loc[ft, features], y_top[ft],
                   eval_set=[(d.loc[vt, features], y_top[vt])], verbose=False)
        y_ext_fit = y_ext[fit_m]
        spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
        stage1 = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                               min_child_weight=8, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=5.0,
                               objective="binary:logistic",
                               scale_pos_weight=spw, eval_metric="aucpr")
        stage1.fit(d.loc[fit_m, features], y_ext_fit,
                   eval_set=[(d.loc[val_m, features], y_ext[val_m])],
                   verbose=False)

    p_ext = stage1.predict_proba(d.loc[te_m, features])[:, 1]
    p_dir = stage2.predict_proba(d.loc[te_m, features])[:, 1]
    te = pd.DataFrame({"s": p_ext * (2 * p_dir - 1),
                       "y": y3[te_m].to_numpy(),
                       "rel": d.loc[te_m, "rel"].to_numpy()},
                      index=d.index[te_m])

    # matched-rate calls: top t% -> over, bottom t% -> under
    s_pct = te.groupby(level="date")["s"].rank(pct=True)
    pred = np.select([s_pct <= t, s_pct >= 1 - t], [0, 2], default=1)
    te["pred"] = pred

    mf1 = f1_score(te["y"], pred, average="macro")
    f1_under, f1_neu, f1_over = f1_score(te["y"], pred, average=None,
                                         labels=[0, 1, 2])
    prec_over = precision_score(te["y"], pred, labels=[2], average=None,
                                zero_division=0)[0]
    prec_under = precision_score(te["y"], pred, labels=[0], average=None,
                                 zero_division=0)[0]

    # money: long overs, short unders, per formation day
    calls = te[te["pred"] != 1].copy()
    sign = np.where(calls["pred"] == 2, 1.0, -1.0)
    calls["pnl_gross"] = sign * calls["rel"]
    calls["pnl_net"] = calls["pnl_gross"] - COST_RT
    day_pnl = calls.groupby(level="date")[["pnl_gross", "pnl_net"]].mean()
    n_per_day = calls.groupby(level="date").size().mean()
    ann = np.sqrt(252 / HORIZON)
    sharpe_net = float(day_pnl["pnl_net"].mean() / day_pnl["pnl_net"].std()
                       * ann)
    ann_ret_net = float(day_pnl["pnl_net"].mean() * 252 / HORIZON)
    ann_ret_gross = float(day_pnl["pnl_gross"].mean() * 252 / HORIZON)
    hit = float((calls["pnl_gross"] > 0).mean())

    row = {"split": f"{int((1-2*t)*100)}/{int(t*100)}/{int(t*100)}",
           "tail_pct": t, "calls_per_day": round(float(n_per_day), 1),
           "macro_f1": round(float(mf1), 4),
           "f1_under": round(float(f1_under), 4),
           "f1_neutral": round(float(f1_neu), 4),
           "f1_over": round(float(f1_over), 4),
           "prec_over": round(float(prec_over), 4),
           "prec_under": round(float(prec_under), 4),
           "hit_rate": round(hit, 4),
           "gross_per_call_10d": round(float(calls["pnl_gross"].mean()), 5),
           "net_per_call_10d": round(float(calls["pnl_net"].mean()), 5),
           "ann_ret_gross": round(ann_ret_gross, 4),
           "ann_ret_net": round(ann_ret_net, 4),
           "sharpe_net": round(sharpe_net, 3)}
    rows.append(row)
    print(f"\n{row['split']}: macro_f1={row['macro_f1']}  "
          f"f1_over={row['f1_over']}  prec_over={row['prec_over']}  "
          f"hit={row['hit_rate']}  net/call={row['net_per_call_10d']}  "
          f"ann_net={row['ann_ret_net']:.1%}  sharpe={row['sharpe_net']}",
          flush=True)

res = pd.DataFrame(rows)
print("\n===== label split sweep (2026, two-stage XGB, matched-rate calls) =====")
print(res.to_string(index=False))
res.to_csv(EXP / "results" / "label_split_sweep.csv", index=False)
print("\nsaved -> experiments/results/label_split_sweep.csv")
