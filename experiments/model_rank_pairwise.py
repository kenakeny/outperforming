"""model_rank_pairwise — XGBoost learning-to-rank instead of classification.

The direction problem may partly be an OBJECTIVE problem: multi-class
log-loss treats every row independently, but the label is inherently
relative ("did this fund beat its peers TODAY"). Learning-to-rank
objectives (rank:pairwise / rank:ndcg) optimize within-day orderings
directly -- each trading day is a query group, relevance = within-day
decile of realized peer-relative excess.

Same 74 features, 2-week horizon, 1y rolling window, 2026 test as the
benchmark ensemble. Judged on the fixed eval protocol.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, gpu_lock
from eval_protocol import evaluate

HORIZON = 10

d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                        window_years=1)
print(f"dataset: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}",
      flush=True)

# relevance grades: within-day decile of rel (0..9, higher = better)
grade = (d.groupby(level="date")["rel"]
         .transform(lambda s: pd.qcut(s.rank(method="first"), 10,
                                      labels=False)).astype("int8"))
qid = d.index.get_level_values("date").factorize()[0]
assert (np.diff(qid) >= 0).all(), "rows must be sorted by day for qid groups"

from xgboost import XGBRanker

for obj in ["rank:pairwise", "rank:ndcg"]:
    with gpu_lock():
        m = XGBRanker(
            objective=obj, n_estimators=3000, learning_rate=0.03,
            max_depth=6, min_child_weight=8, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist",
            device="cuda", eval_metric="ndcg", early_stopping_rounds=150,
            random_state=42, verbosity=0)
        m.fit(d.loc[fit_m, features], grade[fit_m], qid=qid[fit_m],
              eval_set=[(d.loc[val_m, features], grade[val_m])],
              eval_qid=[qid[val_m]], verbose=False)
    score = m.predict(d.loc[te_m, features])
    name = "rank_" + obj.split(":")[1]
    evaluate(name, d, te_m, score,
             extra={"objective": obj, "horizon": HORIZON, "window_years": 1,
                    "best_iteration": int(m.best_iteration)})
