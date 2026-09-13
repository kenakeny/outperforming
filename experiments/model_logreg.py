"""model_logreg — multinomial logistic regression (CPU) on all 9 families.

The linear floor for the sweep: if the boosted trees / LSTM can't beat this,
their extra capacity isn't buying anything. Median-impute + standardize on
train stats only (no test leakage).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from exp_harness import load_sweep_dataset, score_and_save

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
tr_mask = fit_mask | val_mask          # no early stopping -> use all train rows
print(f"dataset: {d.shape}, train={tr_mask.sum()}, test={te_mask.sum()}")

med = d.loc[tr_mask, features].median()
mu = d.loc[tr_mask, features].mean()
sd = d.loc[tr_mask, features].std().replace(0, 1.0)

def prep(mask):
    Z = (d.loc[mask, features].fillna(med) - mu) / sd
    return Z.clip(-5, 5).to_numpy(dtype="float32")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=300, C=1.0, tol=1e-3)
model.fit(prep(tr_mask), d.loc[tr_mask, "target"])

proba = model.predict_proba(prep(te_mask))
pred = proba.argmax(axis=1)
score_and_save("model_logreg", d, te_mask, pred, proba[:, 2],
               extra={"n_features": len(features)})
