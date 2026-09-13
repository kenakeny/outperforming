"""model_xgboost — XGBoost (hist, GPU, gpu-locked) on all 9 feature families."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from exp_harness import load_sweep_dataset, score_and_save, gpu_lock

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
print(f"dataset: {d.shape}, fit={fit_mask.sum()}, val={val_mask.sum()}, test={te_mask.sum()}")

from xgboost import XGBClassifier

with gpu_lock():
    model = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="multi:softprob", num_class=3,
        tree_method="hist", device="cuda", eval_metric="mlogloss",
        early_stopping_rounds=150, random_state=42, verbosity=1)
    model.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
              eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "target"])],
              verbose=200)

proba = model.predict_proba(d.loc[te_mask, features])
pred = proba.argmax(axis=1)
score_and_save("model_xgboost", d, te_mask, pred, proba[:, 2],
               extra={"n_features": len(features),
                      "best_iteration": int(model.best_iteration)})
model.save_model(str(pathlib.Path(__file__).parent / "results" / "model_xgboost.ubj"))
