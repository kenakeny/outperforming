"""model_catboost — CatBoost on all 9 feature families (GPU, gpu-locked).

Uses the best config from exp_09's hyperparameter sweep (depth 10, lr 0.02,
l2 9 — best rank IC and tied-best macro F1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from exp_harness import load_sweep_dataset, score_and_save, gpu_lock

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
print(f"dataset: {d.shape}, fit={fit_mask.sum()}, val={val_mask.sum()}, test={te_mask.sum()}")

from catboost import CatBoostClassifier

with gpu_lock():
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=10, l2_leaf_reg=9,
        loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
        random_seed=42, task_type="GPU", verbose=200,
        early_stopping_rounds=150, use_best_model=True)
    model.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
              eval_set=(d.loc[val_mask, features], d.loc[val_mask, "target"]))

pred = np.asarray(model.predict(d.loc[te_mask, features])).ravel().astype(int)
proba_up = model.predict_proba(d.loc[te_mask, features])[:, 2]
score_and_save("model_catboost", d, te_mask, pred, proba_up,
               extra={"n_features": len(features),
                      "best_iteration": int(model.get_best_iteration() or 3000)})
model.save_model(str(pathlib.Path(__file__).parent / "results" / "model_catboost.cbm"))
