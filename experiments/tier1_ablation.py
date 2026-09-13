"""tier1_ablation — which Tier-1 family helped and which hurt?

The combined 95-feature retrain FAILED the adoption gate (blend dir_acc
0.544 vs 0.555, rank IC 0.074 vs 0.086, 90/5/5 Sharpe 1.41 vs 1.89) even
though validation improved -- some family is overfitting late-2025 /
misleading 2026. Ablate: the two-stage XGB (fast proxy for the full blend)
trained on base-74 plus each family alone, scored on the fixed protocol.
Adopt only families that improve on base.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, gpu_lock
from eval_protocol import evaluate

HORIZON, TRAIN_THR = 10, 0.10
BASE_FAMILIES = [
    "exp_01_voltail.parquet", "exp_02_peermom.parquet", "exp_03_peerext.parquet",
    "exp_04_liquidity.parquet", "exp_05_betaidio.parquet",
    "exp_06_catcontext.parquet", "exp_07_range.parquet",
    "exp_11_downside.parquet", "exp_12_temporal.parquet",
]
VARIANTS = {
    "abl_base74": [],
    "abl_plus_market": ["exp_14_market_state.parquet"],
    "abl_plus_overnight": ["exp_15_overnight.parquet"],
    "abl_plus_spillover": ["exp_16_spillover.parquet"],
}

from xgboost import XGBClassifier
BASEP = dict(n_estimators=3000, tree_method="hist", device="cuda",
             early_stopping_rounds=150, random_state=42, verbosity=0)
TUNED = dict(max_depth=6, learning_rate=0.03, min_child_weight=8,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)

for name, extra in VARIANTS.items():
    d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(
        horizon=HORIZON, window_years=1, families=BASE_FAMILIES + extra)
    day_pct = d.groupby(level="date")["rel"].rank(pct=True)
    y_top = (day_pct >= 1 - TRAIN_THR).astype("int8")
    y_ext = ((day_pct >= 1 - TRAIN_THR) | (day_pct <= TRAIN_THR)).astype("int8")
    y_ext15 = ((day_pct >= 0.85) | (day_pct <= 0.15)).astype("int8")

    ft, vt = fit_m & (y_ext == 1).values, val_m & (y_ext == 1).values
    with gpu_lock():
        s2 = XGBClassifier(**BASEP, **TUNED, objective="binary:logistic",
                           eval_metric="auc")
        s2.fit(d.loc[ft, features], y_top[ft],
               eval_set=[(d.loc[vt, features], y_top[vt])], verbose=False)
        spw = float((y_ext15[fit_m] == 0).sum() / (y_ext15[fit_m] == 1).sum())
        s1 = XGBClassifier(**BASEP, max_depth=8, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0,
                           objective="binary:logistic", scale_pos_weight=spw,
                           eval_metric="aucpr")
        s1.fit(d.loc[fit_m, features], y_ext15[fit_m],
               eval_set=[(d.loc[val_m, features], y_ext15[val_m])],
               verbose=False)
    score = (s1.predict_proba(d.loc[te_m, features])[:, 1] *
             (2 * s2.predict_proba(d.loc[te_m, features])[:, 1] - 1))
    evaluate(name, d, te_m, score, extra={"n_features": len(features)})
