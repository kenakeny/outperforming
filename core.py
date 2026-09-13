"""
core.py -- the five primitives every experiment needs. Nothing else belongs here.

    panel()       features + label + liquidity, one row per (date, ticker)
    fit()         model factory
    walkforward() purged out-of-sample scores
    backtest()    scores -> portfolio returns
    stats()       returns -> the numbers we always report

Replaces four separate copies of each of these that had drifted apart across
us_backtest.py / news_model.py / news_model_v2.py / saudi_vs_tasi.py.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "notebooks")
import common  # noqa: E402
import etl  # noqa: E402

DROP = {"obv", "dollar_vol_20d"}          # absolute-scale, don't transfer
PRICE_UNIT = ["macd", "macd_signal", "macd_hist", "atr_14"]
LABEL_COLS = ["target", "rel", "fwd_ret", "adv"]


def features(close, high, low, volume, cat=None):
    """44 v3 features, made scale-free. One 'all' peer group unless told otherwise,
    so category-derived columns are universe-relative and comparable across markets."""
    cat = pd.Series("all", index=close.columns) if cat is None else cat
    frames, regime = etl.build_features(close, high, low, volume, cat)
    long = etl.features_to_long(frames, regime)
    px = close.stack(future_stack=True).reindex(long.index)
    for c in PRICE_UNIT:
        long[c] = long[c] / px
    return long.drop(columns=[c for c in DROP if c in long.columns]) \
               .replace([np.inf, -np.inf], np.nan)


def panel(horizon=20, min_adv=0.0, close=None, high=None, low=None, volume=None):
    """Feature panel + universe-relative tercile label + trailing dollar volume.

    Label: excess over the equal-weight universe that day, cut into per-day terciles.
    A flat print (fwd_ret == 0) is not a return, so those rows are dropped.
    """
    if close is None:
        close, high, low, volume, _ = etl._load_panel(etl.load_config())
    f = features(close, high, low, volume)

    fwd = close.shift(-horizon) / close - 1.0
    dv = (close * volume).rolling(20).mean()
    d = f.join(pd.concat({"fwd_ret": fwd.stack(future_stack=True),
                          "adv": dv.stack(future_stack=True)}, axis=1), how="inner")
    d = d.dropna(subset=["fwd_ret", "adv"])
    d = d[d["fwd_ret"] != 0]
    if min_adv:
        d = d[d["adv"] >= min_adv]

    g = d.groupby(level="date")["fwd_ret"]
    d["rel"] = d["fwd_ret"] - g.transform("mean")
    pct = d.groupby(level="date")["rel"].rank(pct=True)
    d["target"] = np.select([pct <= 1 / 3, pct <= 2 / 3], [0, 1], default=2).astype(np.int8)
    return d.sort_index(), close.index


def fit(kind, X, y, seed=0, gpu=True):
    if kind == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                               loss_function="MultiClass", verbose=False,
                               allow_writing_files=False, random_seed=seed,
                               **({"task_type": "GPU", "devices": "0"} if gpu else {}))
    elif kind == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              device="cuda" if gpu else "cpu", num_class=3,
                              objective="multi:softprob", eval_metric="mlogloss",
                              random_state=seed, n_jobs=-1)
    elif kind == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, num_class=3,
                               objective="multiclass", random_state=seed, verbose=-1)
    elif kind == "logreg":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000, random_state=seed))
    else:
        raise ValueError(kind)
    m.fit(X, y)
    return m


def walkforward(d, cols, trading_index, horizon=20, kind="xgboost",
                years=range(2019, 2027), min_train=50_000):
    """Purged expanding-window scores. Purge is `horizon` trading days at every
    split -- the invariant that was violated by the h=25 label bug, so it is
    asserted here rather than assumed (see reports/LABEL_HORIZON_BUG.md)."""
    dd = d.reset_index()
    out = []
    for year in years:
        te = dd[(dd["date"] >= f"{year}-01-01") & (dd["date"] <= f"{year}-12-31")]
        if te.empty:
            continue
        cut = common.purge_cutoff(trading_index, te["date"].min(), horizon)
        common.assert_no_overlap(trading_index, cut, te["date"].min(), horizon)
        tr = dd[dd["date"] <= cut]
        if len(tr) < min_train:
            continue
        p = fit(kind, tr[cols], tr["target"]).predict_proba(te[cols])
        keep = ["date", "ticker", "rel", "fwd_ret", "adv"]
        if "vol_20d" in te.columns:          # trailing vol, for inverse-vol weighting
            keep.append("vol_20d")
        out.append(te[keep].assign(score=p[:, 2] - p[:, 0], year=year))
        print(f"  {kind} {year}: {len(tr):,} -> {len(te):,}", flush=True)
    return pd.concat(out, ignore_index=True)


def backtest(scores, horizon=20, top=0.10, cost_bps=5.0, weight="equal", buffer=0.0):
    """Long-only basket from `scores`, rebalanced every `horizon` days so holds
    don't overlap. Cost is charged on turnover against the previous basket.

    weight: 'equal' | 'score' (score-proportional) | 'invvol' (inverse 20d vol).
    """
    days = np.array(sorted(scores["date"].unique()))[::horizon]
    prev, rows = {}, []
    for d0 in days:
        g = scores[scores["date"] == d0]
        if len(g) < 6:
            continue
        n = max(int(round(len(g) * top)), 3)
        ranked = g.sort_values("score", ascending=False)
        if buffer and prev:
            band = ranked.head(min(int(n * (1 + buffer)), len(ranked)))
            inc = band[band["ticker"].isin(prev)]
            pick = pd.concat([inc, ranked[~ranked["ticker"].isin(prev)]
                              .head(max(n - len(inc), 0))]).head(n)
        else:
            pick = ranked.head(n)

        if weight == "score":
            w = (pick["score"] - pick["score"].min() + 1e-9)
        elif weight == "invvol":
            # trailing realized vol only -- weighting by anything derived from
            # fwd_ret would be lookahead dressed up as risk management
            if "vol_20d" not in pick.columns:
                raise KeyError("invvol weighting needs a trailing 'vol_20d' column")
            w = 1.0 / pick["vol_20d"].clip(lower=1e-4)
        else:
            w = pd.Series(1.0, index=pick.index)
        w = w / w.sum()
        held = dict(zip(pick["ticker"], w))

        turn = sum(abs(held.get(t, 0) - prev.get(t, 0))
                   for t in set(held) | set(prev))
        prev = held
        port, bench = float((pick["fwd_ret"] * w).sum()), g["fwd_ret"].mean()
        cost = turn * cost_bps / 1e4
        rows.append({"date": d0, "port": port, "bench": bench, "turn": turn,
                     "cost": cost, "excess_net": port - bench - cost, "n": len(pick)})
    return pd.DataFrame(rows)


def stats(bt, horizon=20, col=None):
    """The numbers we always report. `col` defaults to net excess over the universe."""
    py = 252 / horizon
    r = bt[col] if col else bt["port"] - bt["bench"] - bt["cost"]
    sd = r.std()
    return {"n": len(r), "ann": float(r.mean() * py),
            "vol": float(sd * np.sqrt(py)),
            "ir": float(r.mean() * py / (sd * np.sqrt(py))) if sd else np.nan,
            "t": float(r.mean() / (sd / np.sqrt(len(r)))) if sd else np.nan,
            "hit": float((r > 0).mean()), "turn": float(bt["turn"].mean())}
