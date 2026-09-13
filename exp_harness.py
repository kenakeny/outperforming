"""
exp_harness.py — shared data / label / eval harness for the 10-experiment sweep.

Every experiment script in experiments/ imports from here so the label,
embargo, holdout, and metrics CANNOT drift between concurrently-working
agents (same reason notebooks/common.py exists). Typical experiment:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from exp_harness import (load_universe, stack, assemble, loo_peer_mean,
                             cat_pct_rank, build_label, save_features,
                             run_experiment)

    fields, cat = load_universe()
    close = fields["Close"]
    ... build wide (date x ticker) feature frames, strictly trailing (<= t) ...
    X   = assemble({"name": frame, ...})
    lbl = build_label(close, cat)
    save_features("exp_01_voltail", X)
    run_experiment("exp_01_voltail", X, lbl, close.index)

GPU training is serialized through a lockfile (experiments/gpu.lock) so
parallel agents don't fight over the card. Feature building runs in parallel.

Label + holdout are identical to train_5d.py: next-5-trading-day forward
return, leave-one-out category-peer mean -> peer-relative excess -> per-day
terciles; 5-day embargo; TEST_YEAR held out.
"""
import json
import os
import pathlib
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import yaml

HORIZON, SKIP, MIN_GROUP, TEST_YEAR = 5, 0, 3, 2026
EMBARGO = HORIZON + SKIP
TASK_TYPE = "GPU"

ROOT = pathlib.Path(__file__).resolve().parent
cfg  = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RAW  = ROOT / cfg["paths"]["raw"]
PROC = ROOT / cfg["paths"]["processed"]
EXP  = ROOT / "experiments"
(EXP / "features").mkdir(parents=True, exist_ok=True)
(EXP / "results").mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS = dict(
    iterations=2000, learning_rate=0.04, depth=7, l2_leaf_reg=3,
    loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
    random_seed=42, task_type=TASK_TYPE, verbose=200,
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_universe():
    """OHLCV wide frames restricted to funds in a real category
    (>= MIN_GROUP peers, not 'Uncategorized'). Returns (fields, cat) where
    fields = {'Open'|'High'|'Low'|'Close'|'Volume': wide date x ticker frame}."""
    md = pd.read_parquet(RAW / "market_data.parquet")
    md.index = pd.to_datetime(md.index); md.index.name = "date"
    meta = pd.read_parquet(RAW / "metadata.parquet")
    cat = meta["category"].reindex(md["Close"].columns)
    counts = cat.value_counts()
    keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
    kept = cat[cat.isin(keep)].index
    fields = {}
    for f in ["Open", "High", "Low", "Close", "Volume"]:
        w = md[f][kept]; w.columns.name = "ticker"; fields[f] = w
    return fields, cat[kept]


def stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s


def assemble(feat_dict):
    """dict[name] -> wide frame  ==>  long (date, ticker) design matrix."""
    return pd.DataFrame({k: stack(v) for k, v in feat_dict.items()})


def loo_peer_mean(x, cat):
    """Leave-one-out category-peer mean of wide frame x, per day (a fund is
    never part of its own benchmark)."""
    s = x.T.groupby(cat).transform("sum").T
    n = x.notna().T.groupby(cat).transform("sum").T
    return (s - x.fillna(0)) / (n - x.notna().astype(int)).replace(0, np.nan)


def cat_pct_rank(x, cat):
    """Cross-sectional percentile rank (0..1) of x within category, per day."""
    return x.T.groupby(cat).rank(pct=True).T


# ---------------------------------------------------------------------------
# label (identical to train_5d.py)
# ---------------------------------------------------------------------------

def build_label_h(close, cat, horizon, skip=0):
    """Peer-relative tercile at an arbitrary forward horizon. Returns
    DataFrame with columns target (int8 0/1/2), rel (continuous peer-rel
    excess), fwd_ret, indexed by (date, ticker). The horizon-parameterized
    generalization of build_label -- previously copy-pasted (as
    build_label_h) across the horizon_2week / combined_best / fix_flip_hard
    / final_matrices scripts."""
    fwd = close.shift(-horizon) / close.shift(-skip) - 1.0
    rel = fwd - loo_peer_mean(fwd, cat)
    fwd_l, rel_l = stack(fwd), stack(rel)
    target = (rel_l.dropna().groupby(level="date")
              .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                           labels=[0, 1, 2])).astype("int8"))
    return (pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1)
            .dropna(subset=["target"]))


def build_label(close, cat):
    """Next-HORIZON-day peer-relative tercile (the default 5d label)."""
    return build_label_h(close, cat, HORIZON, SKIP)


# ---------------------------------------------------------------------------
# GPU lock — serialize training across concurrent agent processes
# ---------------------------------------------------------------------------

_GPU_LOCK = EXP / "gpu.lock"

@contextmanager
def gpu_lock(timeout_s=7200, stale_s=1800, poll_s=10):
    start = time.time()
    while True:
        try:
            fd = os.open(_GPU_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(_GPU_LOCK) > stale_s:
                    os.remove(_GPU_LOCK)          # holder likely crashed
                    continue
            except OSError:
                continue                           # lock vanished between checks
            if time.time() - start > timeout_s:
                raise TimeoutError("timed out waiting for GPU lock")
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            os.remove(_GPU_LOCK)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# holdout + eval
# ---------------------------------------------------------------------------

def rolling_window_masks(date_index, trading_index, test_year=TEST_YEAR,
                         years=1, embargo=EMBARGO):
    """(fit, val, test) masks like holdout_masks, but the train window is a
    ROLLING window of `years` calendar years ending at the embargoed cutoff,
    instead of expanding back to the start of the data. The window-size sweep
    (validate_window.py) showed 1y rolling beats full history; this was then
    copy-pasted (as window_masks) across 5 later experiment scripts -- now it
    lives here. `embargo` must match the label horizon (HORIZON + SKIP)."""
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    cut = trading_index[max(trading_index.searchsorted(test_start) - embargo, 0)]
    start = test_start - pd.DateOffset(years=years)
    tr = (date_index <= cut) & (date_index >= start)
    te = (date_index >= test_start) & (date_index <= test_end)
    tr_dates = np.sort(date_index[tr].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.90)]
    val_cut = trading_index[max(trading_index.searchsorted(val_start) - embargo, 0)]
    fit = tr & (date_index <= val_cut)
    val = tr & (date_index >= val_start)
    return fit, val, te


def holdout_masks(date_index, trading_index, test_year=TEST_YEAR,
                  embargo=EMBARGO):
    """(fit, val, test) boolean masks. Train <= test_year start minus `embargo`
    trading days; last 10% of train dates = early-stopping val (also embargoed);
    test = calendar test_year only. `embargo` must match the label horizon."""
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end   = pd.Timestamp(f"{test_year}-12-31")
    cut = trading_index[max(trading_index.searchsorted(test_start) - embargo, 0)]
    tr_mask = date_index <= cut
    te_mask = (date_index >= test_start) & (date_index <= test_end)
    tr_dates = np.sort(date_index[tr_mask].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.90)]
    val_cut = trading_index[max(trading_index.searchsorted(val_start) - embargo, 0)]
    fit_mask = tr_mask & (date_index <= val_cut)
    val_mask = tr_mask & (date_index >= val_start)
    return fit_mask, val_mask, te_mask


def run_experiment(name, X, lbl, trading_index, features=None, params=None,
                   test_year=TEST_YEAR):
    """Join features to label, train a CatBoost holdout model (GPU-locked),
    save importance + metrics under experiments/results/, return the metrics
    dict. `X` long (date,ticker) design matrix; `lbl` from build_label()."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, f1_score

    features = features or list(X.columns)
    params = {**DEFAULT_PARAMS, **(params or {})}

    d = X.join(lbl, how="inner").dropna(subset=["target"])
    d = d.dropna(subset=features, how="all")
    d["target"] = d["target"].astype("int8")
    date_index = d.index.get_level_values("date")
    fit_mask, val_mask, te_mask = holdout_masks(date_index, trading_index, test_year)

    with gpu_lock():
        model = CatBoostClassifier(early_stopping_rounds=120, use_best_model=True,
                                   **params)
        model.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
                  eval_set=(d.loc[val_mask, features], d.loc[val_mask, "target"]))

    y_te  = d.loc[te_mask, "target"]
    pred  = np.asarray(model.predict(d.loc[te_mask, features])).ravel().astype(int)
    proba = model.predict_proba(d.loc[te_mask, features])[:, 2]
    ic = (pd.DataFrame({"s": proba, "rel": d.loc[te_mask, "rel"]}, index=y_te.index)
          .groupby(level="date")
          .apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean())

    imp = (pd.Series(model.get_feature_importance(), index=features)
           .sort_values(ascending=False))
    imp.rename("importance").to_csv(EXP / "results" / f"{name}_importance.csv",
                                    header=True)
    res = {
        "name": name, "test_year": int(test_year),
        "n_features": len(features), "n_train": int(fit_mask.sum()),
        "n_test": int(te_mask.sum()),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "rank_ic": round(float(ic), 4),
        "best_iteration": int(model.get_best_iteration() or params["iterations"]),
        "features": features,
    }
    (EXP / "results" / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(f"\n[{name}] macro F1={res['macro_f1']}  acc={res['accuracy']}  "
          f"rankIC={res['rank_ic']}  (n_test={res['n_test']})")
    print("top importance:\n" + imp.head(10).round(2).to_string())
    return res


def save_features(name, X):
    """Persist the long design matrix so exp_10 can combine all families."""
    path = EXP / "features" / f"{name}.parquet"
    X.to_parquet(path)
    print("features saved ->", path)


# ---------------------------------------------------------------------------
# model sweep — shared loader + scorer so every model (catboost / xgboost /
# lightgbm / logistic / lstm) trains on the identical design matrix and is
# scored with identical metrics. Model-specific code lives in the
# experiments/model_*.py scripts only.
# ---------------------------------------------------------------------------

SWEEP_FAMILIES = [
    "exp_01_voltail.parquet", "exp_02_peermom.parquet", "exp_03_peerext.parquet",
    "exp_04_liquidity.parquet", "exp_05_betaidio.parquet",
    "exp_06_catcontext.parquet", "exp_07_range.parquet",
    "exp_11_downside.parquet", "exp_12_temporal.parquet",
    "exp_15_overnight.parquet",
]

# Tier-1 ablation (tier1_ablation.py, 2026): exp_15_overnight cleared the
# adoption gate (rank IC 0.0876 vs 0.0822 base, K2 payoff +23%, w/l 1.35);
# exp_14_market_state badly hurt direction (broadcast day-level features let
# trees memorize regimes -> dir_acc 0.512) and exp_16_spillover lost money
# at the gate -- both deliberately excluded. Their parquets remain in
# experiments/features/ for opt-in use via load_sweep_dataset(families=...).

# exp_13_composition (static yfinance holdings snapshot) is deliberately NOT
# in the default sweep: strong standalone (F1 0.4719) but redundant with the
# price features for boosted trees (xgb F1 0.5016 -> 0.4992, rank IC 0.026 ->
# 0.021 when added; only the linear model improved). Opt in via
# load_sweep_dataset(families=SWEEP_FAMILIES + ["exp_13_composition.parquet"]).


def load_sweep_dataset(test_year=TEST_YEAR, families=None, horizon=None,
                       window_years=None):
    """All feature families + label, float32, plus the holdout masks.
    Returns (d, features, fit_mask, val_mask, te_mask, trading_index).

    horizon: label horizon in trading days (default HORIZON=5; pass 10 for
    the 2-week label). The embargo scales with it automatically.
    window_years: if set, use a rolling `window_years`-year training window
    (rolling_window_masks) instead of the expanding-window holdout_masks."""
    frames = [pd.read_parquet(EXP / "features" / f)
              for f in (families or SWEEP_FAMILIES)
              if (EXP / "features" / f).exists()]
    X = pd.concat(frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated(keep="first")].astype("float32")
    # ratio features can emit +/-inf on zero denominators; xgboost hard-errors
    # on inf and inf corrupts mean/std standardization -> treat as missing
    X = X.mask(np.isinf(X))
    features = list(X.columns)

    fields, cat = load_universe()
    close = fields["Close"]
    h = HORIZON if horizon is None else horizon
    lbl = build_label_h(close, cat, h, SKIP)

    d = X.join(lbl, how="inner").dropna(subset=["target"])
    d = d.dropna(subset=features, how="all")
    d["target"] = d["target"].astype("int8")
    date_index = d.index.get_level_values("date")
    embargo = h + SKIP
    if window_years is not None:
        fit_mask, val_mask, te_mask = rolling_window_masks(
            date_index, close.index, test_year, window_years, embargo)
    else:
        fit_mask, val_mask, te_mask = holdout_masks(date_index, close.index,
                                                    test_year, embargo)
    return d, features, fit_mask, val_mask, te_mask, close.index


def score_and_save(name, d, te_mask, pred, proba_up, extra=None,
                   test_year=TEST_YEAR):
    """Score test-year predictions (macro F1 / accuracy / daily rank IC of
    P(outperform) vs realized peer-relative excess) and write the metrics
    json under experiments/results/. Same metrics as run_experiment."""
    from sklearn.metrics import accuracy_score, f1_score
    y_te = d.loc[te_mask, "target"]
    pred = np.asarray(pred).ravel().astype(int)
    ic = (pd.DataFrame({"s": np.asarray(proba_up, dtype=float),
                        "rel": d.loc[te_mask, "rel"]}, index=y_te.index)
          .groupby(level="date")
          .apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean())
    res = {
        "name": name, "test_year": int(test_year),
        "n_test": int(te_mask.sum()),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "rank_ic": round(float(ic), 4),
        **(extra or {}),
    }
    (EXP / "results" / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(f"\n[{name}] macro F1={res['macro_f1']}  acc={res['accuracy']}  "
          f"rankIC={res['rank_ic']}  (n_test={res['n_test']})")
    return res
