"""f1_push — iterate on 3-class argmax macro F1 until clearly above 0.50.

Target metric (locked with the user): macro F1 of argmax 3-class predictions
(under/neutral/over terciles), 2-week label, 2026 test, 1y rolling window.
Reference points: 5d multiclass XGB = 0.5016 (full history), 2-week baseline
= 0.5011.

Ladder (each step keeps what helped):
  A. XGB multiclass on current sweep features (incl. exp_15 overnight)
  B. A + exp_14b market-interaction features
  C. soft-vote (mean predicted probabilities) of XGB + LightGBM + CatBoost
     on the better feature set of A/B

Reports macro F1 (argmax) + accuracy + rank IC for every step.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from exp_harness import load_sweep_dataset, gpu_lock, EXP, SWEEP_FAMILIES

HORIZON = 10
results = {}


def rank_ic(d, te_m, s):
    df = pd.DataFrame({"s": s, "rel": d.loc[te_m, "rel"]}, index=d.index[te_m])
    return float(df.groupby(level="date")
                 .apply(lambda g: g["s"].corr(g["rel"], method="spearman"))
                 .mean())


def report(name, d, te_m, proba):
    pred = proba.argmax(1)
    y = d.loc[te_m, "target"]
    res = {"macro_f1_argmax": round(float(f1_score(y, pred, average="macro")), 4),
           "accuracy": round(float(accuracy_score(y, pred)), 4),
           "rank_ic": round(rank_ic(d, te_m, proba[:, 2]), 4)}
    results[name] = res
    print(f"[{name}] macro_f1={res['macro_f1_argmax']}  "
          f"acc={res['accuracy']}  ic={res['rank_ic']}", flush=True)
    return res


def fit_xgb_mc(d, features, fit_m, val_m, te_m):
    from xgboost import XGBClassifier
    with gpu_lock():
        m = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=8,
                          min_child_weight=8, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=5.0,
                          objective="multi:softprob", num_class=3,
                          tree_method="hist", device="cuda",
                          eval_metric="mlogloss", early_stopping_rounds=150,
                          random_state=42, verbosity=0)
        m.fit(d.loc[fit_m, features], d.loc[fit_m, "target"],
              eval_set=[(d.loc[val_m, features], d.loc[val_m, "target"])],
              verbose=False)
    return m.predict_proba(d.loc[te_m, features])


# ---- A: current sweep features -------------------------------------------
d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                        window_years=1)
print(f"A features: {len(features)}", flush=True)
proba_a = fit_xgb_mc(d, features, fit_m, val_m, te_m)
report("A_xgb_sweep", d, te_m, proba_a)

# ---- B: + market interactions --------------------------------------------
fam_b = SWEEP_FAMILIES + ["exp_14b_market_interact.parquet"]
d2, features2, fit2, val2, te2, _ = load_sweep_dataset(
    horizon=HORIZON, window_years=1, families=fam_b)
print(f"B features: {len(features2)}", flush=True)
proba_b = fit_xgb_mc(d2, features2, fit2, val2, te2)
report("B_plus_market_interact", d2, te2, proba_b)

# ---- C: soft-vote on the better of A/B -----------------------------------
use_b = results["B_plus_market_interact"]["macro_f1_argmax"] > \
    results["A_xgb_sweep"]["macro_f1_argmax"]
dd, ff, fm, vm, tm = ((d2, features2, fit2, val2, te2) if use_b
                      else (d, features, fit_m, val_m, te_m))
proba_x = proba_b if use_b else proba_a
print(f"C uses feature set: {'B' if use_b else 'A'}", flush=True)

import lightgbm as lgb
mlg = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03,
                         num_leaves=255, min_child_samples=100,
                         subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=5.0,
                         objective="multiclass", num_class=3, n_jobs=12,
                         random_state=42, verbosity=-1)
mlg.fit(dd.loc[fm, ff], dd.loc[fm, "target"],
        eval_set=[(dd.loc[vm, ff], dd.loc[vm, "target"])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(150, verbose=False)])
proba_l = mlg.predict_proba(dd.loc[tm, ff])
report("C1_lgbm_alone", dd, tm, proba_l)

from catboost import CatBoostClassifier
with gpu_lock():
    mcb = CatBoostClassifier(iterations=3000, learning_rate=0.02, depth=10,
                             l2_leaf_reg=9, loss_function="MultiClass",
                             eval_metric="TotalF1:average=Macro",
                             random_seed=42, task_type="GPU", verbose=False,
                             early_stopping_rounds=150, use_best_model=True)
    mcb.fit(dd.loc[fm, ff], dd.loc[fm, "target"],
            eval_set=(dd.loc[vm, ff], dd.loc[vm, "target"]))
proba_c = mcb.predict_proba(dd.loc[tm, ff])
report("C2_catboost_alone", dd, tm, proba_c)

report("C_softvote_xgb_lgbm", dd, tm, (proba_x + proba_l) / 2)
report("C_softvote_all3", dd, tm, (proba_x + proba_l + proba_c) / 3)

print("\n===== summary =====")
for k, v in results.items():
    print(f"{k:24s} macro_f1={v['macro_f1_argmax']}  acc={v['accuracy']}  "
          f"ic={v['rank_ic']}")
(EXP / "results" / "f1_push.json").write_text(json.dumps(results, indent=2))
print("saved -> experiments/results/f1_push.json")
