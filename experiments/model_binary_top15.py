"""model_binary_top15 — binary XGBoost: top-15% outperformers vs the rest.

Positive class = fund's peer-relative forward-5d excess (`rel`) in the top
15% of its trading day; everything else negative. Class imbalance (15/85)
is offset with scale_pos_weight = n_neg/n_pos on the fit set. Same
74-feature matrix, embargo, and 2026 holdout as the multiclass sweep
(loader/masks from exp_harness), so results are directly comparable.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, gpu_lock, EXP

TOP_PCT = 0.15

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()

# per-day percentile of peer-relative excess -> top 15% = positive
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y"] = (day_pct >= 1.0 - TOP_PCT).astype("int8")

y_fit = d.loc[fit_mask, "y"]
spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
print(f"dataset: {d.shape}  fit={fit_mask.sum()}  val={val_mask.sum()}  "
      f"test={te_mask.sum()}  pos_rate(fit)={y_fit.mean():.3f}  "
      f"scale_pos_weight={spw:.2f}")

from xgboost import XGBClassifier

with gpu_lock():
    model = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="binary:logistic",
        scale_pos_weight=spw, tree_method="hist", device="cuda",
        eval_metric="aucpr", early_stopping_rounds=150,
        random_state=42, verbosity=1)
    model.fit(d.loc[fit_mask, features], y_fit,
              eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "y"])],
              verbose=200)

# ---------------------------------------------------------------- evaluate
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

y_te = d.loc[te_mask, "y"].to_numpy()
proba = model.predict_proba(d.loc[te_mask, features])[:, 1]
pred = (proba >= 0.5).astype(int)

# rank IC: does the score order funds by realized peer-relative excess?
ic = (pd.DataFrame({"s": proba, "rel": d.loc[te_mask, "rel"]},
                   index=d.index[te_mask])
      .groupby(level="date")
      .apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean())

# practical use: each day take the model's top-15% highest scores -> what
# fraction are true top-15% outperformers? (base rate = 0.15)
te = pd.DataFrame({"s": proba, "y": y_te}, index=d.index[te_mask])
prec_at_15 = (te.groupby(level="date")
              .apply(lambda g: g.nlargest(max(int(len(g) * TOP_PCT), 1), "s")["y"].mean())
              .mean())

res = {
    "name": "model_binary_top15", "test_year": 2026,
    "n_test": int(te_mask.sum()),
    "pos_rate_test": round(float(y_te.mean()), 4),
    "scale_pos_weight": round(spw, 2),
    "accuracy": round(float(accuracy_score(y_te, pred)), 4),
    "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 4),
    "precision_pos": round(float(precision_score(y_te, pred)), 4),
    "recall_pos": round(float(recall_score(y_te, pred)), 4),
    "f1_pos": round(float(f1_score(y_te, pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
    "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
    "precision_at_top15_daily": round(float(prec_at_15), 4),
    "rank_ic": round(float(ic), 4),
    "best_iteration": int(model.best_iteration),
    "n_features": len(features),
}
(EXP / "results" / "model_binary_top15.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

imp = (pd.Series(model.feature_importances_, index=features)
       .sort_values(ascending=False))
imp.rename("importance").to_csv(EXP / "results" / "model_binary_top15_importance.csv",
                                header=True)
print("\ntop importance:\n" + imp.head(10).round(4).to_string())
