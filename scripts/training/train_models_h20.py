"""The 20-trading-day-horizon counterpart to
train_models.py's "+sr" models.

Same purged walk-forward, same three model families (LogReg / XGBoost /
CatBoost), same +sr feature set (44 v3 features + 11 support/resistance) --
the only things that change are the dataset (dataset_v3_h20.parquet, built by
build_dataset_h20.py) and the purge width (20 trading days instead of 5,
matching the label's own horizon). +sr+news and the LSTM are skipped here:
this is meant to sit *alongside* the existing 5-day production models as a
second selectable horizon, not to reproduce the full model-comparison sweep.

Saved artifacts (models/):
  catboost_sr_20d.cbm  xgboost_sr_20d.ubj  logreg_sr_20d.joblib
  results_h20.json / results_h20.md
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import common

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

HORIZON = 20
TEST_YEARS = list(range(2019, 2027))
LABEL = "target"
CLIP = 10.0  # S/R ratio features have rare tiny-denominator outliers
LOGREG_MAX_ROWS = 500_000

RANDOM_MACRO_F1 = 0.3335  # uniform-random baseline on a balanced 3-class target


def load_data():
    d = pd.read_parquet("data/processed/dataset_v3_h20.parquet")
    base_cols = [c for c in d.columns if c not in ("target", "fwd_ret", "rel")]

    sr = pd.read_parquet("data/processed/sr_features.parquet")
    sr_cols = list(sr.columns)
    d = d.join(sr, how="left")

    feat = base_cols + sr_cols
    d[feat] = d[feat].replace([np.inf, -np.inf], np.nan)
    d[sr_cols] = d[sr_cols].clip(-CLIP, CLIP)

    d = d.reset_index().dropna(subset=[LABEL])
    d[LABEL] = d[LABEL].astype(int)
    return d, feat


def fit_xgb(train, cols):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
        objective="multi:softprob", num_class=3, n_jobs=-1, eval_metric="mlogloss")
    clf.fit(train[cols], train[LABEL])
    return clf


def fit_logreg(train, cols):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train) > LOGREG_MAX_ROWS:
        train = train.sample(LOGREG_MAX_ROWS, random_state=0)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000))
    pipe.fit(train[cols], train[LABEL])
    return pipe


def fit_cat(train, cols):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
        verbose=False, allow_writing_files=False, task_type="GPU", devices="0")
    clf.fit(train[cols], train[LABEL])
    return clf


def save_model(model, path):
    if hasattr(model, "save_model"):
        model.save_model(str(path))
    else:
        import joblib
        joblib.dump(model, path)


def summarize(name, metrics, pooled):
    if metrics.empty:
        print(f"\n### {name}: no folds"); return None
    acc = accuracy_score(pooled["label"], pooled["pred"])
    mf1 = f1_score(pooled["label"], pooled["pred"], average="macro")
    print(f"\n### {name}")
    print(metrics.to_string(index=False))
    print(f"  POOLED: accuracy={acc:.4f}  macro_f1={mf1:.4f}")
    return {"accuracy": acc, "macro_f1": mf1}


def main():
    print(f"loading dataset_v3_h20 (horizon={HORIZON}) + S/R...")
    d, cols = load_data()
    trading_index = pd.read_parquet("data/raw/prices.parquet").sort_index().index
    print(f"rows={len(d):,} features={len(cols)}")

    results = {}
    for name, fit_fn, ext in [("LogReg", fit_logreg, "joblib"),
                              ("XGBoost", fit_xgb, "ubj"), ("CatBoost", fit_cat, "cbm")]:
        t0 = time.time()
        m, pooled, last_model = common.walk_forward(
            d, cols, fit_fn, TEST_YEARS, HORIZON, trading_index, label_col=LABEL)
        res = summarize(f"{name} [+sr, h={HORIZON}]", m, pooled)
        results[f"{name.lower()}_sr_20d"] = res
        path = MODELS / f"{name.lower()}_sr_20d.{ext}"
        save_model(last_model, path)
        print(f"  saved {path.name}  ({time.time()-t0:.0f}s)")

    json.dump(results, open(MODELS / "results_h20.json", "w"), indent=2)
    lines = ["# Model results (v3 + S/R, 20-trading-day horizon)\n",
             "| Model | macro-F1 | accuracy |", "|---|---|---|"]
    for k, v in results.items():
        if v:
            lines.append(f"| {k} | {v['macro_f1']:.4f} | {v['accuracy']:.4f} |")
    lines.append(f"\nBaseline: uniform random macro-F1 {RANDOM_MACRO_F1:.4f}. "
                 f"Label = 20-day peer-relative tercile, purged {HORIZON} trading days.")
    (MODELS / "results_h20.md").write_text("\n".join(lines))

    print("\n" + "=" * 48 + "\nSUMMARY (pooled macro-F1)")
    print(f"  {'uniform random':24} {RANDOM_MACRO_F1:.4f}  <- beat this")
    for k, v in results.items():
        if v:
            print(f"  {k:24} {v['macro_f1']:.4f}")
    print(f"\nmodels + results saved to {MODELS}")


if __name__ == "__main__":
    main()
