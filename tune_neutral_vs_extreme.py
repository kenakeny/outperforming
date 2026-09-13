"""tune_neutral_vs_extreme.py -- Optuna search over XGBoost hyperparameters for
the 15/70/15 neutral-vs-extreme binary model. Optimizes validation AUC (the
early-stopping split from holdout_masks, NOT the test year -- the test year
stays untouched until the final refit so the search itself can't leak into it).
"""
import pathlib

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from exp_harness import gpu_lock
from model_neutral_vs_extreme_15_70_15 import load_dataset_15_70_15

OUT = pathlib.Path("reports/model_eval")
N_TRIALS = 40

d, features, fit_mask, val_mask, te_mask = load_dataset_15_70_15()
d["is_extreme"] = (d["target"] != 1).astype("int8")

X_fit, y_fit = d.loc[fit_mask, features], d.loc[fit_mask, "is_extreme"]
X_val, y_val = d.loc[val_mask, features], d.loc[val_mask, "is_extreme"]
X_te, y_te = d.loc[te_mask, features], d.loc[te_mask, "is_extreme"]

BASELINE_AUC = 0.8624   # from the hand-picked-params run, for comparison


def objective(trial):
    params = dict(
        n_estimators=3000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        objective="binary:logistic", tree_method="hist", device="cuda",
        eval_metric="auc", early_stopping_rounds=100, random_state=42, verbosity=0,
    )
    with gpu_lock():
        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    trial.set_user_attr("best_iteration", int(model.best_iteration))
    return auc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nbest val AUC: {study.best_value:.4f}  (baseline hand-picked params: {BASELINE_AUC:.4f})")
    print("best params:", study.best_params)

    best = {**study.best_params, "n_estimators": 3000,
            "objective": "binary:logistic", "tree_method": "hist", "device": "cuda",
            "eval_metric": "auc", "early_stopping_rounds": 100, "random_state": 42,
            "verbosity": 0}
    with gpu_lock():
        final = XGBClassifier(**best)
        final.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    proba_te = final.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    from sklearn.metrics import f1_score, accuracy_score
    test_auc = roc_auc_score(y_te, proba_te)
    test_f1 = f1_score(y_te, pred_te, average="macro")
    test_acc = accuracy_score(y_te, pred_te)
    print(f"\nTEST YEAR -- tuned: auc={test_auc:.4f} macro_f1={test_f1:.4f} acc={test_acc:.4f}")
    print(f"TEST YEAR -- baseline (hand-picked): auc=0.8624 macro_f1=0.7552 acc=0.7991")

    final.save_model(str(OUT.parent.parent / "models" /
                         "model_xgboost_neutral_vs_extreme_15_70_15_tuned.ubj"))
    study.trials_dataframe().to_csv(OUT / "optuna_neutral_vs_extreme_trials.csv", index=False)
    pd.DataFrame([{"trial": t.number, "value": t.value, **t.params}
                 for t in study.trials]).to_csv(
        OUT / "optuna_neutral_vs_extreme_trials_readable.csv", index=False)
    print(f"\nsaved tuned model + trial logs -> {OUT}/")
