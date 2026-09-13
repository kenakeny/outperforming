"""model_neutral_vs_extreme_15_70_15.py -- same neutral-vs-extreme reframing as
model_neutral_vs_extreme.py, but with an asymmetric label split: bottom 15% of
the daily peer-relative excess-return distribution = "under", top 15% =
"over", middle 70% = "neutral" (vs. the tercile 33/33/33 split used elsewhere).
Extreme = under OR over. Same features, same embargo, same holdout as
model_xgboost.py.
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

from exp_harness import (load_universe, stack, loo_peer_mean, holdout_masks,
                         gpu_lock, HORIZON, SKIP, TEST_YEAR, EXP, SWEEP_FAMILIES)

OUT = pathlib.Path("reports/model_eval")
OUT.mkdir(parents=True, exist_ok=True)
CUTS = (0.15, 0.85)   # bottom 15% / middle 70% / top 15%


def build_label_split(close, cat, horizon, skip, cuts):
    fwd = close.shift(-horizon) / close.shift(-skip) - 1.0
    rel = fwd - loo_peer_mean(fwd, cat)
    fwd_l, rel_l = stack(fwd), stack(rel)
    pct = rel_l.dropna().groupby(level="date").rank(pct=True)
    target = pd.Series(1, index=pct.index, dtype="int8")   # neutral
    target[pct <= cuts[0]] = 0                             # under
    target[pct > cuts[1]] = 2                              # over
    return (pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1)
            .dropna(subset=["target"]))


def load_dataset_15_70_15():
    frames = [pd.read_parquet(EXP / "features" / f) for f in SWEEP_FAMILIES
              if (EXP / "features" / f).exists()]
    X = pd.concat(frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated(keep="first")].astype("float32")
    X = X.mask(np.isinf(X))
    features = list(X.columns)

    fields, cat = load_universe()
    close = fields["Close"]
    lbl = build_label_split(close, cat, HORIZON, SKIP, CUTS)

    d = X.join(lbl, how="inner").dropna(subset=["target"])
    d = d.dropna(subset=features, how="all")
    d["target"] = d["target"].astype("int8")
    date_index = d.index.get_level_values("date")
    embargo = HORIZON + SKIP
    fit_mask, val_mask, te_mask = holdout_masks(date_index, close.index, TEST_YEAR, embargo)
    return d, features, fit_mask, val_mask, te_mask


d, features, fit_mask, val_mask, te_mask = load_dataset_15_70_15()
d["is_extreme"] = (d["target"] != 1).astype("int8")

print(f"dataset: {d.shape}, fit={fit_mask.sum()}, val={val_mask.sum()}, test={te_mask.sum()}")
vc = d.loc[fit_mask, "target"].value_counts(normalize=True).sort_index()
print(f"class balance (train): under={vc.get(0,0):.3f}  neutral={vc.get(1,0):.3f}  over={vc.get(2,0):.3f}")
print(f"binary balance (train): extreme={d.loc[fit_mask,'is_extreme'].mean():.3f}  "
      f"neutral={1 - d.loc[fit_mask,'is_extreme'].mean():.3f}")

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

model.save_model(str(OUT.parent.parent / "models" / "model_xgboost_neutral_vs_extreme_15_70_15.ubj"))

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

tag = "xgboost_us_neutral_vs_extreme_15_70_15"

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, label=f"extreme vs neutral (AUC={auc_:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("XGBoost US -- neutral vs extreme (15/70/15) -- ROC", fontsize=10)
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / f"{tag}_roc.png", dpi=150); plt.close(fig)

cm = confusion_matrix(y_true, pred, labels=[0, 1], normalize="true")
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["neutral", "extreme"]).plot(
    ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
ax.set_title("XGBoost US -- neutral vs extreme (15/70/15)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"{tag}_confusion.png", dpi=150); plt.close(fig)

imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
imp.to_csv(OUT / f"{tag}_importance.csv", header=["importance"])
top = imp.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1))
ax.barh(top.index, top.values)
ax.set_title("XGBoost US -- neutral vs extreme (15/70/15) -- feature importance", fontsize=10)
ax.set_xlabel("gain-based importance")
fig.tight_layout(); fig.savefig(OUT / f"{tag}_importance.png", dpi=150); plt.close(fig)

pd.DataFrame([{"name": "XGBoost US (neutral vs extreme, 15/70/15)", "macro_f1": macro_f1,
              "accuracy": acc, "auc": auc_, "n_test": int(te_mask.sum()),
              "n_features": len(features),
              "best_iteration": int(model.best_iteration)}]).to_csv(
    OUT / "neutral_vs_extreme_15_70_15_summary.csv", index=False)
print(f"\nsaved figures + summary -> {OUT}/")
