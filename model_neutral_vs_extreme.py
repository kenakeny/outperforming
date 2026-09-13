"""model_neutral_vs_extreme.py -- reframes the 3-class tercile label (under/
neutral/over) as a binary problem: does this fund end up in a tail bucket
(under OR over) 5 days out, or does it stay near its peer-relative neutral
band? Same features, same holdout, same embargo as model_xgboost.py --
only the label collapses classes {0, 2} -> 1 ("extreme"), {1} -> 0 ("neutral").
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, auc, f1_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

from exp_harness import load_sweep_dataset, gpu_lock

OUT = pathlib.Path("reports/model_eval")
OUT.mkdir(parents=True, exist_ok=True)

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
d["is_extreme"] = (d["target"] != 1).astype("int8")

print(f"dataset: {d.shape}, fit={fit_mask.sum()}, val={val_mask.sum()}, test={te_mask.sum()}")
print(f"class balance (train): extreme={d.loc[fit_mask, 'is_extreme'].mean():.3f}  "
      f"neutral={1 - d.loc[fit_mask, 'is_extreme'].mean():.3f}")

with gpu_lock():
    model = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="binary:logistic",
        tree_method="hist", device="cuda", eval_metric="auc",
        early_stopping_rounds=150, random_state=42, verbosity=1)
    model.fit(d.loc[fit_mask, features], d.loc[fit_mask, "is_extreme"],
              eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "is_extreme"])],
              verbose=200)

model.save_model(str(OUT.parent.parent / "models" / "model_xgboost_neutral_vs_extreme.ubj"))

te = d.loc[te_mask]
proba = model.predict_proba(te[features])[:, 1]
pred = (proba >= 0.5).astype(int)
y_true = te["is_extreme"].to_numpy()

macro_f1 = f1_score(y_true, pred, average="macro")
acc = accuracy_score(y_true, pred)
fpr, tpr, _ = roc_curve(y_true, proba)
auc_ = auc(fpr, tpr)
print(f"\nmacro_f1={macro_f1:.4f}  acc={acc:.4f}  auc={auc_:.4f}  "
      f"best_iteration={model.best_iteration}")

# ---- ROC -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, label=f"extreme vs neutral (AUC={auc_:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("XGBoost US -- neutral vs extreme -- ROC"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "xgboost_us_neutral_vs_extreme_roc.png", dpi=150)
plt.close(fig)

# ---- confusion matrix --------------------------------------------------------
cm = confusion_matrix(y_true, pred, labels=[0, 1], normalize="true")
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["neutral", "extreme"]).plot(
    ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
ax.set_title("XGBoost US -- neutral vs extreme -- confusion (row-normalized)")
fig.tight_layout(); fig.savefig(OUT / "xgboost_us_neutral_vs_extreme_confusion.png", dpi=150)
plt.close(fig)

# ---- feature importance ------------------------------------------------------
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
imp.to_csv(OUT / "xgboost_us_neutral_vs_extreme_importance.csv", header=["importance"])
top = imp.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1))
ax.barh(top.index, top.values)
ax.set_title("XGBoost US -- neutral vs extreme -- feature importance")
ax.set_xlabel("gain-based importance")
fig.tight_layout(); fig.savefig(OUT / "xgboost_us_neutral_vs_extreme_importance.png", dpi=150)
plt.close(fig)

pd.DataFrame([{"name": "XGBoost US (neutral vs extreme)", "macro_f1": macro_f1,
              "accuracy": acc, "auc": auc_, "n_test": int(te_mask.sum()),
              "n_features": len(features),
              "best_iteration": int(model.best_iteration)}]).to_csv(
    OUT / "neutral_vs_extreme_summary.csv", index=False)
print(f"\nsaved figures + summary -> {OUT}/")
