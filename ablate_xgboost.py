"""ablate_xgboost.py -- retrain the XGBoost US model with the top-N most
important features removed (per reports/model_eval/xgboost_us_importance.csv)
and compare against the full-feature baseline.

    python ablate_xgboost.py --top-n 5
"""
import argparse
import pathlib

import pandas as pd
from xgboost import XGBClassifier

from exp_harness import load_sweep_dataset, gpu_lock
from eval_report import report, importance_plot, OUT


def train_and_eval(d, features, fit_mask, val_mask, te_mask, tag, title):
    with gpu_lock():
        model = XGBClassifier(
            n_estimators=3000, learning_rate=0.03, max_depth=8,
            min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="multi:softprob", num_class=3,
            tree_method="hist", device="cuda", eval_metric="mlogloss",
            early_stopping_rounds=150, random_state=42, verbosity=0)
        model.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
                  eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "target"])],
                  verbose=False)

    te = d.loc[te_mask]
    proba = model.predict_proba(te[features])
    pred = proba.argmax(axis=1)
    y_true = te["target"].to_numpy()
    scores = (te.reset_index()[["date", "ticker", "fwd_ret", "rel"]]
              .assign(score=proba[:, 2] - proba[:, 0], adv=1.0))
    importance_plot(model, features, f"{title} -- feature importance",
                    OUT / f"{tag}_importance.png")
    res = report(title, y_true, pred, proba[:, 2], scores, horizon=5, tag=tag)
    res["n_features"] = len(features)
    res["best_iteration"] = int(model.best_iteration)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=5,
                   help="number of top-importance features to drop")
    a = p.parse_args()

    d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()

    imp = pd.read_csv(OUT / "xgboost_us_importance.csv", index_col=0)["importance"]
    dropped = list(imp.sort_values(ascending=False).head(a.top_n).index)
    print(f"dropping top {a.top_n} features: {dropped}")
    ablated_features = [f for f in features if f not in dropped]
    print(f"features: {len(features)} -> {len(ablated_features)}")

    baseline = train_and_eval(d, features, fit_mask, val_mask, te_mask,
                              "xgboost_us_full", "XGBoost US (full features)")
    ablated = train_and_eval(d, ablated_features, fit_mask, val_mask, te_mask,
                             f"xgboost_us_ablated_top{a.top_n}",
                             f"XGBoost US (top-{a.top_n} features removed)")
    ablated["dropped_features"] = ",".join(dropped)

    df = pd.DataFrame([baseline, ablated])
    df.to_csv(OUT / f"ablation_top{a.top_n}_summary.csv", index=False)
    print(f"\n{'metric':12} {'full':>10} {'ablated':>10} {'delta':>10}")
    for k in ("macro_f1", "auc", "ann", "ir", "t"):
        f, ab = baseline[k], ablated[k]
        print(f"{k:12} {f:>10.4f} {ab:>10.4f} {ab - f:>+10.4f}")


if __name__ == "__main__":
    main()
