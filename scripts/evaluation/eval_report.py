"""Generate a backtest curve, ROC (outperform-vs-rest), macro F1,
and confusion matrix for the XGBoost US model and the Saudi transfer arms.

    python -m scripts.evaluation.eval_report

Writes PNGs to reports/model_eval/.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (RocCurveDisplay, roc_curve, auc, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

import core

OUT = pathlib.Path("reports/model_eval")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["under", "neutral", "over"]


def roc_plot(y_true, proba_over, title, path):
    fpr, tpr, _ = roc_curve((y_true == 2).astype(int), proba_over)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"outperform vs rest (AUC={a:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return a


def confusion_plot(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues",
                                                            values_format=".2f",
                                                            colorbar=False)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def backtest_plot(scores, horizon, title, path, cost_bps=15.0, top=0.33):
    bt = core.backtest(scores, horizon=horizon, top=top, cost_bps=cost_bps)
    st = core.stats(bt, horizon=horizon)
    cum_port = (1 + bt["port"] - bt["cost"]).cumprod()
    cum_bench = (1 + bt["bench"]).cumprod()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bt["date"], cum_port, label="model portfolio", lw=2)
    ax.plot(bt["date"], cum_bench, label="universe (equal-weight)", lw=2, ls="--")
    ax.set_title(f"{title}\nIR={st['ir']:+.2f}  ann.excess={st['ann']:+.2%}  t={st['t']:+.2f}")
    ax.set_ylabel("growth of $1"); ax.legend(); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return st


def report(name, y_true, y_pred, proba_over, scores_df, horizon, tag):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auc_ = roc_plot(y_true, proba_over, f"{name} -- ROC (outperform vs rest)",
                    OUT / f"{tag}_roc.png")
    confusion_plot(y_true, y_pred, f"{name} -- confusion matrix (row-normalized)",
                   OUT / f"{tag}_confusion.png")
    st = backtest_plot(scores_df, horizon, f"{name} -- backtest", OUT / f"{tag}_backtest.png")
    print(f"\n[{name}] macro_f1={macro_f1:.4f}  auc={auc_:.4f}  "
          f"ann.excess={st['ann']:+.2%}  IR={st['ir']:+.2f}  t={st['t']:+.2f}")
    return {"name": name, "macro_f1": macro_f1, "auc": auc_, **st}


def importance_plot(model, features, title, path, top_n=25):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    imp.to_csv(path.with_suffix(".csv"), header=["importance"])
    top = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1))
    ax.barh(top.index, top.values)
    ax.set_title(title); ax.set_xlabel("gain-based importance")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return imp


def xgboost_us():
    from exp_harness import load_sweep_dataset
    from xgboost import XGBClassifier

    d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
    model = XGBClassifier()
    model.load_model("models/model_xgboost.ubj")

    te = d.loc[te_mask]
    proba = model.predict_proba(te[features])
    pred = proba.argmax(axis=1)
    y_true = te["target"].to_numpy()

    scores = (te.reset_index()[["date", "ticker", "fwd_ret", "rel"]]
              .assign(score=proba[:, 2] - proba[:, 0], adv=1.0))
    importance_plot(model, features, "XGBoost (US ETFs) -- feature importance",
                    OUT / "xgboost_us_importance.png")
    return report("XGBoost (US ETFs)", y_true, pred, proba[:, 2], scores,
                  horizon=5, tag="xgboost_us")


def saudi(arm="finetune"):
    p = pd.read_parquet("reports/saudi_predictions.parquet").loc[arm]
    y_true = p["target"].to_numpy()
    pred = p["pred"].to_numpy()
    # score in [-2, 2] (proba_over - proba_under); rescale to a pseudo-proba in [0,1]
    proba_over = (p["score"] - p["score"].min()) / (p["score"].max() - p["score"].min())
    scores = p[["date", "ticker", "fwd_ret", "rel", "adv"]].assign(score=p["score"])
    return report(f"Saudi transfer ({arm})", y_true, pred, proba_over, scores,
                  horizon=20, tag=f"saudi_{arm}")


if __name__ == "__main__":
    rows = [xgboost_us(),
            saudi("finetune"), saudi("saudi_only"), saudi("zeroshot")]
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"\nsaved figures + summary.csv -> {OUT}/")
