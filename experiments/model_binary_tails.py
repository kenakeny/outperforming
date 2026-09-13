"""model_binary_tails — binary XGBoost: top-15% vs bottom-15% outperformers.

Follow-up to model_binary_top15, which learned "volatile fund" rather than
direction (idio_vol carried 53% importance, rank IC ~0) because high-vol
funds sit in BOTH tails. Here the middle 70% is dropped entirely: positive =
top 15% of daily peer-relative excess, negative = bottom 15%. Classes are
balanced by construction, so no class weighting. Volatility can no longer
separate the classes — only directional signal can.

Same 74 features, embargo, and 2026 holdout as the sweep. Scored two ways:
on the tails themselves (the trained task) and as a daily ranking signal
across ALL test funds (rank IC), which is the fair apples-to-apples number
against the multiclass models.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, gpu_lock, EXP

TOP_PCT = 0.15

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()

day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y"] = (day_pct >= 1.0 - TOP_PCT).astype("int8")
tail = (day_pct >= 1.0 - TOP_PCT) | (day_pct <= TOP_PCT)

fit_t, val_t, te_t = fit_mask & tail, val_mask & tail, te_mask & tail
print(f"tail rows: fit={fit_t.sum()}  val={val_t.sum()}  test={te_t.sum()}  "
      f"pos_rate(fit)={d.loc[fit_t, 'y'].mean():.3f}")

from xgboost import XGBClassifier

with gpu_lock():
    model = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="binary:logistic",
        tree_method="hist", device="cuda", eval_metric="auc",
        early_stopping_rounds=150, random_state=42, verbosity=1)
    model.fit(d.loc[fit_t, features], d.loc[fit_t, "y"],
              eval_set=[(d.loc[val_t, features], d.loc[val_t, "y"])],
              verbose=200)

# ---------------------------------------------------------------- evaluate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

# (a) the trained task: separate the two tails in 2026
y_te = d.loc[te_t, "y"].to_numpy()
proba_t = model.predict_proba(d.loc[te_t, features])[:, 1]
pred_t = (proba_t >= 0.5).astype(int)

# (b) ranking signal across ALL 2026 funds (comparable to the sweep's rank IC)
proba_all = model.predict_proba(d.loc[te_mask, features])[:, 1]
all_te = pd.DataFrame({"s": proba_all, "rel": d.loc[te_mask, "rel"],
                       "y": d.loc[te_mask, "y"]}, index=d.index[te_mask])
ic_all = (all_te.groupby(level="date")
          .apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean())
prec_at_15 = (all_te.groupby(level="date")
              .apply(lambda g: g.nlargest(max(int(len(g) * TOP_PCT), 1), "s")["y"].mean())
              .mean())

res = {
    "name": "model_binary_tails", "test_year": 2026,
    "n_test_tails": int(te_t.sum()),
    "accuracy": round(float(accuracy_score(y_te, pred_t)), 4),
    "precision_pos": round(float(precision_score(y_te, pred_t)), 4),
    "recall_pos": round(float(recall_score(y_te, pred_t)), 4),
    "f1_pos": round(float(f1_score(y_te, pred_t)), 4),
    "roc_auc": round(float(roc_auc_score(y_te, proba_t)), 4),
    "rank_ic_all_funds": round(float(ic_all), 4),
    "precision_at_top15_daily": round(float(prec_at_15), 4),
    "best_iteration": int(model.best_iteration),
    "n_features": len(features),
}
(EXP / "results" / "model_binary_tails.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

imp = (pd.Series(model.feature_importances_, index=features)
       .sort_values(ascending=False))
imp.rename("importance").to_csv(EXP / "results" / "model_binary_tails_importance.csv",
                                header=True)
print("\ntop importance:\n" + imp.head(10).round(4).to_string())
