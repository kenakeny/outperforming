"""model_extreme_f1 — how high can macro F1 honestly go on the task the data
actually supports: extreme (either tail) vs neutral, binary.

The 3-class direction task is information-limited (~0.50 argmax macro F1
ceiling this session). Extremeness detection is the strong signal in this
data (top15-vs-rest hit ROC AUC 0.80). This trains extreme(30%)-vs-
neutral(70%) as a plain binary classifier and tunes the decision threshold
ON VALIDATION ONLY to maximize binary macro F1, then reports test macro F1
once. 2-week horizon, 1y rolling window, 2026 test.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             confusion_matrix)
from exp_harness import load_sweep_dataset, gpu_lock, EXP

HORIZON, THR = 10, 0.15

d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                        window_years=1)
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
y_ext = ((day_pct >= 1 - THR) | (day_pct <= THR)).astype("int8")
print(f"fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}  "
      f"pos_rate={y_ext[fit_m].mean():.3f}", flush=True)

from xgboost import XGBClassifier
with gpu_lock():
    m = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="binary:logistic", tree_method="hist",
        device="cuda", eval_metric="auc", early_stopping_rounds=150,
        random_state=42, verbosity=0)
    m.fit(d.loc[fit_m, features], y_ext[fit_m],
          eval_set=[(d.loc[val_m, features], y_ext[val_m])], verbose=False)

p_val = m.predict_proba(d.loc[val_m, features])[:, 1]
p_te = m.predict_proba(d.loc[te_m, features])[:, 1]
y_val, y_te = y_ext[val_m].to_numpy(), y_ext[te_m].to_numpy()

# threshold tuned on validation only
ths = np.arange(0.05, 0.95, 0.01)
f1s = [f1_score(y_val, (p_val >= t).astype(int), average="macro") for t in ths]
best_t = float(ths[int(np.argmax(f1s))])
print(f"threshold tuned on val: {best_t:.2f} (val macro F1 {max(f1s):.4f})")

pred = (p_te >= best_t).astype(int)
cm = confusion_matrix(y_te, pred)
res = {
    "name": "extreme_vs_neutral_binary", "horizon": HORIZON,
    "window_years": 1, "threshold": best_t,
    "test_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
    "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
    "test_roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
    "confusion_matrix": cm.tolist(),
    "best_iteration": int(m.best_iteration),
}
print(json.dumps({k: v for k, v in res.items() if k != "confusion_matrix"},
                 indent=2))
print(pd.DataFrame(cm, index=["neutral", "extreme"],
                   columns=["neutral", "extreme"]).to_string())
cmn = cm / cm.sum(axis=1, keepdims=True)
print("row-normalized:")
print(pd.DataFrame(cmn.round(3), index=["neutral", "extreme"],
                   columns=["neutral", "extreme"]).to_string())
(EXP / "results" / "model_extreme_f1.json").write_text(json.dumps(res, indent=2))
