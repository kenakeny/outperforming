"""model_lightgbm — LightGBM (CPU) on all 9 feature families.

Runs on CPU deliberately so it can train concurrently with the GPU-locked
models; num_threads capped to leave cores for the other CPU jobs.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from exp_harness import load_sweep_dataset, score_and_save

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
print(f"dataset: {d.shape}, fit={fit_mask.sum()}, val={val_mask.sum()}, test={te_mask.sum()}")

import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=3000, learning_rate=0.03, num_leaves=255, max_depth=-1,
    min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=5.0, objective="multiclass",
    num_class=3, n_jobs=8, random_state=42)
model.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
          eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "target"])],
          eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])

proba = model.predict_proba(d.loc[te_mask, features])
pred = proba.argmax(axis=1)
score_and_save("model_lightgbm", d, te_mask, pred, proba[:, 2],
               extra={"n_features": len(features),
                      "best_iteration": int(model.best_iteration_ or 3000)})
