"""quickwins_direction — Tier-1 quick wins #2 and #5 from DIRECTION_FIX_PLAN.

Four stage2 (direction) variants on the standard 2-week / 1y-window setup,
all sharing ONE stage1 extreme gate so the comparison isolates stage2:

  a. baseline           — current production stage2 (reproduction check)
  b. magnitude-weighted — sample_weight = |rel| in stage2 fit (big moves
                          teach more than barely-extreme ones)
  c. normalized target  — stage2's training label built from rel / same-day
                          category dispersion (regime-invariant target; the
                          2022 fix). Evaluation unchanged (standard rel).
  d. b + c combined

Each scored on the fixed eval protocol (2026).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, load_universe, gpu_lock
from eval_protocol import evaluate

HORIZON, TRAIN_THR = 10, 0.10

d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                        window_years=1)
print(f"fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}", flush=True)

_, cat = load_universe()
d["cat"] = d.index.get_level_values("ticker").map(cat)

# standard target percentiles (raw rel) and normalized variant
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
cat_disp = (d.groupby([pd.Grouper(level="date"), d["cat"]])["rel"]
            .transform("std").replace(0, np.nan))
rel_norm = d["rel"] / cat_disp
day_pct_norm = rel_norm.groupby(level="date").rank(pct=True)

from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, tree_method="hist", device="cuda",
            early_stopping_rounds=150, random_state=42, verbosity=0)
TUNED = dict(max_depth=6, learning_rate=0.03, min_child_weight=8,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)

# one shared stage1 (extreme gate on standard 15% definition)
y_ext_15 = ((day_pct >= 0.85) | (day_pct <= 0.15)).astype("int8")
spw = float((y_ext_15[fit_m] == 0).sum() / (y_ext_15[fit_m] == 1).sum())
with gpu_lock():
    stage1 = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0,
                           objective="binary:logistic", scale_pos_weight=spw,
                           eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], y_ext_15[fit_m],
               eval_set=[(d.loc[val_m, features], y_ext_15[val_m])],
               verbose=False)
p_ext = stage1.predict_proba(d.loc[te_m, features])[:, 1]
print("shared stage1 trained", flush=True)


def run_variant(name, pct_series, weighted):
    y_top = (pct_series >= 1 - TRAIN_THR).astype("int8")
    y_ext = ((pct_series >= 1 - TRAIN_THR) |
             (pct_series <= TRAIN_THR)).astype("int8")
    ft, vt = fit_m & (y_ext == 1).values, val_m & (y_ext == 1).values
    sw = d.loc[ft, "rel"].abs().to_numpy() if weighted else None
    with gpu_lock():
        m = XGBClassifier(**BASE, **TUNED, objective="binary:logistic",
                          eval_metric="auc")
        m.fit(d.loc[ft, features], y_top[ft], sample_weight=sw,
              eval_set=[(d.loc[vt, features], y_top[vt])], verbose=False)
    p_dir = m.predict_proba(d.loc[te_m, features])[:, 1]
    score = p_ext * (2.0 * p_dir - 1.0)
    return evaluate(name, d, te_m, score,
                    extra={"weighted": weighted,
                           "target": ("normalized" if pct_series is day_pct_norm
                                      else "raw"),
                           "best_iteration": int(m.best_iteration)})


run_variant("qw_a_baseline", day_pct, False)
run_variant("qw_b_magweight", day_pct, True)
run_variant("qw_c_normtarget", day_pct_norm, False)
run_variant("qw_d_both", day_pct_norm, True)
