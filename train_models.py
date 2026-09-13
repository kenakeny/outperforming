"""
train_models.py -- XGBoost / CatBoost / LSTM on dataset_v3 augmented with
support/resistance (build_sr_features.py) and FinBERT news (score_sentiment.py).

Consolidated trainer (supersedes the earlier train_news_models.py). Runs the
repo's purged walk-forward (common.walk_forward, horizon 5, test years 2019-26),
reports per-year + pooled macro-F1 / accuracy, and SAVES every fitted model to
models/ plus a metrics summary.

Feature sets compared:
  +sr        : 44 v3 features + 11 support/resistance
  +sr+news   : the above + 4 news-sentiment features
(base = v3 only; reference numbers from the previous run are printed for context.)

Saved artifacts (models/):
  xgb_<set>.ubj  cat_<set>.cbm  lstm_sr_news.pt (+ lstm_sr_news_meta.json)
  results.json / results.md
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "notebooks")
import common

ROOT = pathlib.Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

HORIZON = 5
TEST_YEARS = list(range(2019, 2027))
LABEL = "target"
NEWS_COLS = ["news_sent_1d", "news_n_1d", "news_sent_5d", "news_n_5d"]
CLIP = 10.0  # S/R ratio features have rare tiny-denominator outliers
LOGREG_MAX_ROWS = 500_000  # cap on the linear baseline's training sample per fold

# Reference baselines, measured on the corrected 5-day label over the 2019-2026 test
# rows (see reports/LABEL_HORIZON_BUG.md). macro-F1 on a balanced 3-class target:
#   uniform random    0.3335   <- the number to beat
#   always-majority   0.1696   (accuracy 0.3411)
# The old "v3-only reference" constants (XGBoost 0.4336 / CatBoost 0.4360) were
# produced from the leaky dataset and have been removed rather than kept as a
# misleading comparison.
RANDOM_MACRO_F1 = 0.3335
BASE_REF = {}


def load_data():
    d = pd.read_parquet("data/processed/dataset_v3.parquet")
    base_cols = [c for c in d.columns if c not in ("target", "fwd_ret", "rel")]

    sr = pd.read_parquet("data/processed/sr_features.parquet")
    sr_cols = list(sr.columns)
    d = d.join(sr, how="left")

    news = pd.read_parquet("data/processed/news_features.parquet")
    d = d.join(news, how="left")
    d[NEWS_COLS] = d[NEWS_COLS].fillna(0.0)

    feat = base_cols + sr_cols + NEWS_COLS
    d[feat] = d[feat].replace([np.inf, -np.inf], np.nan)
    d[sr_cols] = d[sr_cols].clip(-CLIP, CLIP)

    d = d.reset_index().dropna(subset=[LABEL])
    d[LABEL] = d[LABEL].astype(int)
    return d, base_cols, sr_cols


def fit_xgb(train, cols):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
        objective="multi:softprob", num_class=3, n_jobs=-1, eval_metric="mlogloss")
    clf.fit(train[cols], train[LABEL])
    return clf


def fit_logreg(train, cols):
    """Multinomial logistic regression -- the linear baseline every tree model is
    judged against. Median-imputed and standardized inside the pipeline, because
    unlike the boosters, logreg has no native NaN handling and is scale-sensitive.
    Subsampled to keep the L-BFGS fit to minutes rather than hours; the point is a
    reference number, not a tuned competitor."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train) > LOGREG_MAX_ROWS:
        train = train.sample(LOGREG_MAX_ROWS, random_state=0)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        # lbfgs on a 3-class target is multinomial by default; the explicit
        # multi_class kwarg is deprecated in sklearn 1.5+ and removed in 1.8
        LogisticRegression(max_iter=1000),
    )
    pipe.fit(train[cols], train[LABEL])
    return pipe


def fit_cat(train, cols):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
        verbose=False, allow_writing_files=False, task_type="GPU", devices="0")
    clf.fit(train[cols], train[LABEL])
    return clf


def run_lstm(d, cols, trading_index, seq_len=20, epochs=4, max_train_seq=300_000):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self, nf):
            super().__init__()
            self.lstm = nn.LSTM(nf, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])

    rows, pooled, last = [], [], None
    for year in TEST_YEARS:
        ts, te_ = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        te_mask = (d["date"] >= ts) & (d["date"] <= te_)
        if not te_mask.any():
            continue
        first_test = d.loc[te_mask, "date"].min()
        cut = common.purge_cutoff(trading_index, first_test, HORIZON)
        common.assert_no_overlap(trading_index, cut, first_test, HORIZON)

        tr_mask = d["date"] <= cut
        mu = d.loc[tr_mask, cols].mean()
        sd = d.loc[tr_mask, cols].std().replace(0, 1)

        tr_starts, te_starts, blocks, labs, off = [], [], [], [], 0
        for _, g in d.groupby("ticker", sort=False):
            g = g.sort_values("date")
            f = np.nan_to_num(((g[cols] - mu) / sd).to_numpy(np.float32))
            lab = g[LABEL].to_numpy(np.int64)
            dts = g["date"].to_numpy()
            blocks.append(f)
            labs.append(lab)
            for j in range(len(g) - seq_len + 1):
                ed = dts[j + seq_len - 1]
                gs = off + j
                if ed <= cut:
                    tr_starts.append(gs)
                elif ts <= ed <= te_:
                    te_starts.append(gs)
            off += len(g)

        feats = torch.from_numpy(np.concatenate(blocks)).to(dev)
        labels = torch.from_numpy(np.concatenate(labs)).to(dev)
        rng = np.random.default_rng(0)
        if len(tr_starts) > max_train_seq:
            tr_starts = rng.choice(tr_starts, max_train_seq, replace=False)
        if not len(te_starts):
            continue

        def batches(starts, bs, shuffle):
            starts = np.asarray(starts)
            if shuffle:
                rng.shuffle(starts)
            for s in range(0, len(starts), bs):
                idx = starts[s:s + bs]
                X = torch.stack([feats[i:i + seq_len] for i in idx])
                yield X, labels[idx + seq_len - 1]

        net = Net(len(cols)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        net.train()
        for _ in range(epochs):
            for X, y in batches(tr_starts, 512, True):
                opt.zero_grad(); lossf(net(X), y).backward(); opt.step()

        net.eval()
        preds, ys = [], []
        with torch.no_grad():
            for X, y in batches(te_starts, 1024, False):
                preds.append(net(X).argmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
        pred, ytrue = np.concatenate(preds), np.concatenate(ys)
        rows.append({"test_year": year, "n_test": len(ytrue),
                     "accuracy": accuracy_score(ytrue, pred),
                     "macro_f1": f1_score(ytrue, pred, average="macro")})
        pooled.append(pd.DataFrame({"label": ytrue, "pred": pred}))
        last = (net, mu, sd)
        print(f"    LSTM {year}: acc={rows[-1]['accuracy']:.4f} mf1={rows[-1]['macro_f1']:.4f}", flush=True)

    metrics = pd.DataFrame(rows)
    pool = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    return metrics, pool, last


def save_model(model, path):
    """Boosters serialize themselves; the sklearn pipeline needs joblib."""
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
    print("loading dataset_v3 + S/R + news...")
    d, base_cols, sr_cols = load_data()
    sets = {"+sr": base_cols + sr_cols, "+sr+news": base_cols + sr_cols + NEWS_COLS}
    trading_index = pd.read_parquet("data/raw/prices.parquet").sort_index().index
    print(f"rows={len(d)} base={len(base_cols)} sr={len(sr_cols)} news={len(NEWS_COLS)}")

    results = {}

    for name, fit_fn, ext in [("LogReg", fit_logreg, "joblib"),
                              ("XGBoost", fit_xgb, "ubj"), ("CatBoost", fit_cat, "cbm")]:
        if name in BASE_REF:
            print(f"\n(reference) {name} base pooled macro_f1 = {BASE_REF[name]:.4f}")
        for tag, cols in sets.items():
            t0 = time.time()
            m, pooled, last_model = common.walk_forward(
                d, cols, fit_fn, TEST_YEARS, HORIZON, trading_index, label_col=LABEL)
            res = summarize(f"{name} [{tag}]", m, pooled)
            results[f"{name} {tag}"] = res
            path = MODELS / f"{name.lower()}_{tag.strip('+').replace('+', '_')}.{ext}"
            save_model(last_model, path)
            print(f"  saved {path.name}  ({time.time()-t0:.0f}s)")

    print("\n=== LSTM [+sr+news] (20d lookback) ===")
    t0 = time.time()
    cols = sets["+sr+news"]
    m, pooled, last = run_lstm(d, cols, trading_index)
    results["LSTM +sr+news"] = summarize("LSTM [+sr+news]", m, pooled)
    if last is not None:
        import torch
        net, mu, sd = last
        torch.save(net.state_dict(), MODELS / "lstm_sr_news.pt")
        json.dump({"features": cols, "seq_len": 20,
                   "mean": mu.tolist(), "std": sd.tolist()},
                  open(MODELS / "lstm_sr_news_meta.json", "w"), indent=2)
        print(f"  saved lstm_sr_news.pt (+meta)  ({time.time()-t0:.0f}s)")

    # persist results
    json.dump(results, open(MODELS / "results.json", "w"), indent=2)
    lines = ["# Model results (v3 + S/R + news)\n",
             "| Model | macro-F1 | accuracy |", "|---|---|---|"]
    for k, v in results.items():
        if v:
            lines.append(f"| {k} | {v['macro_f1']:.4f} | {v['accuracy']:.4f} |")
    lines.append(f"\nBaseline: uniform random macro-F1 {RANDOM_MACRO_F1:.4f} "
                 "(always-majority 0.1696). Label = 5-day peer-relative tercile, "
                 "purged 5 trading days; see reports/LABEL_HORIZON_BUG.md.")
    (MODELS / "results.md").write_text("\n".join(lines))

    print("\n" + "=" * 48 + "\nSUMMARY (pooled macro-F1)")
    print(f"  {'uniform random':24} {RANDOM_MACRO_F1:.4f}  <- beat this")
    for k, v in results.items():
        if v:
            print(f"  {k:24} {v['macro_f1']:.4f}")
    print(f"\nmodels + results saved to {MODELS}")


if __name__ == "__main__":
    main()
