"""serve/saudi.py -- the Saudi (Tadawul) transfer-learning results, served.

There is no live model here the way `inference.py` has one. `saudi.py` trains a
fresh LSTM per purged walk-forward fold as part of evaluating whether a US-
trained net transfers to 12 Tadawul-listed funds (9400.SR-9411.SR) -- it never
persists a single artifact, so there is nothing to load and score today's date
with. What *is* servable is the historical record that evaluation already
wrote: `reports/saudi_predictions.parquet`, one row per (arm, date, ticker)
with the model's score alongside the forward return that already happened.

That distinction matters and the API says so explicitly (`is_realized_history`
on every response): unlike `/signals`, a Saudi "latest" row is not an
actionable call on the future -- `fwd_ret`/`target` are the outcome the score
was graded against, already known when this file was written. This module is
the walk-forward record made explorable, not a second live signal.

Arms: saudi_only (trained on Saudi alone) / zeroshot (US-trained net, Saudi-
fold feature scaling) / zeroshot_pure (US-trained net, US feature scaling --
no Saudi data touches the model at all) / finetune (US-pretrained, head
retrained on Saudi). See saudi.py's own docstring for the transfer design.
"""
import functools
import json
import pathlib

import numpy as np
import pandas as pd

import core

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRED_PATH = ROOT / "reports" / "saudi_predictions.parquet"
NAMES_PATH = ROOT / "data" / "raw" / "saudi_fund_names.json"
BENCH_CACHE = ROOT / "data" / "processed" / "transfer" / "saudi_benchmark_cache.json"

ARMS = ("saudi_only", "zeroshot", "zeroshot_pure", "finetune")
CLASS_NAMES = {0: "underperform", 1: "neutral", 2: "outperform"}
HORIZON = 20      # trading days -- the label horizon saudi.py scored these predictions against
COST_BPS = 15.0   # matches saudi.py's --cost-bps default


@functools.lru_cache(maxsize=1)
def available():
    return PRED_PATH.exists()


@functools.lru_cache(maxsize=1)
def load_predictions():
    """Every (arm, date, ticker) prediction ever written, arm as a column."""
    if not PRED_PATH.exists():
        return pd.DataFrame(columns=["arm", "date", "ticker", "score", "fwd_ret",
                                     "target", "pred"])
    p = pd.read_parquet(PRED_PATH).reset_index(level="arm")
    return p.reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def load_names():
    """Ticker -> fund name, fetched once from yfinance and cached to disk --
    Tadawul funds aren't in financedatabase, which is US-only."""
    if not NAMES_PATH.exists():
        return {}
    raw = json.loads(NAMES_PATH.read_text())
    return {t: v.get("name", t) for t, v in raw.items()}


def arms():
    have = set(load_predictions()["arm"].unique()) if available() else set()
    return [a for a in ARMS if a in have]


def latest_date(arm):
    p = load_predictions()
    p = p[p["arm"] == arm]
    if p.empty:
        raise KeyError(f"no predictions for arm '{arm}'")
    return p["date"].max()


def scored(arm, date=None):
    """Every fund's prediction on `date` (default: latest), ranked by score.

    Columns mirror the US `/signals` shape where the concepts line up
    (ticker, name, score, prediction) and add what only exists here:
    `target`/`fwd_ret`, the realized outcome the row already knows.
    """
    p = load_predictions()
    p = p[p["arm"] == arm]
    if p.empty:
        raise KeyError(f"unknown or unavailable arm '{arm}'; choose one of {ARMS}")

    ts = pd.Timestamp(date) if date else p["date"].max()
    day = p[p["date"] == ts]
    if day.empty:
        raise KeyError(f"no Saudi predictions for {ts.date()} on arm '{arm}'; "
                       f"latest available is {p['date'].max().date()}")

    names = load_names()
    out = day.sort_values("score", ascending=False).copy()
    out["name"] = out["ticker"].map(names).fillna(out["ticker"])
    out["prediction"] = out["pred"].map(CLASS_NAMES)
    out["realized_class"] = out["target"].map(CLASS_NAMES)
    return ts, out[["ticker", "name", "score", "prediction", "realized_class", "fwd_ret"]]


def history(ticker, arm, days=180):
    p = load_predictions()
    p = p[(p["arm"] == arm) & (p["ticker"] == ticker.upper())]
    if p.empty:
        raise KeyError(f"no '{arm}' predictions for '{ticker}'")
    names = load_names()
    out = p.sort_values("date").tail(days).copy()
    out["prediction"] = out["pred"].map(CLASS_NAMES)
    out["realized_class"] = out["target"].map(CLASS_NAMES)
    return {"name": names.get(ticker.upper(), ticker.upper()),
            "rows": out[["date", "score", "prediction", "realized_class", "fwd_ret"]]}


def _fetch_benchmarks(dates):
    """TASI + FALCOM forward returns over `dates` -- the one live network call
    this module makes, so its result is cached to disk (see `benchmark`)."""
    import saudi as saudi_mod  # the research script -- reuses its yfinance call, not a copy of it
    return saudi_mod.benchmarks(dates, HORIZON)


def _stats(returns):
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    py = 252 / HORIZON
    ann = float(r.mean() * py) if len(r) else float("nan")
    vol = float(r.std() * np.sqrt(py)) if len(r) else float("nan")
    return {"ann_return": ann, "ir": ann / vol if vol else None,
            "hit_rate": float((r > 0).mean()) if len(r) else None, "n": int(len(r))}


def benchmark(arm):
    """Model equity curve vs TASI and FALCOM over the same rebalance dates,
    plus ann/IR/hit-rate per series -- the comparison saudi_vs_tasi_plot.py
    draws, served as data instead of a static PNG.

    All four arms are evaluated on the same walk-forward fold boundaries, so
    they share one rebalance-date calendar; the whole disk cache is keyed on
    the predictions file's mtime (not per arm), and the one-time TASI/FALCOM
    network pull is shared across every arm rather than repeated per arm.
    """
    p = load_predictions()
    p = p[p["arm"] == arm]
    if p.empty:
        raise KeyError(f"unknown or unavailable arm '{arm}'; choose one of {ARMS}")

    cache_key = str(PRED_PATH.stat().st_mtime) if PRED_PATH.exists() else "missing"
    cached = _read_bench_cache()
    if cached.get("_key") != cache_key:
        cached = {"_key": cache_key}

    bt = core.backtest(p, horizon=HORIZON, top=0.33, cost_bps=COST_BPS)

    if "_bench" not in cached:
        bench = _fetch_benchmarks(bt["date"])
        cached["_bench"] = {
            name: [None if (v is None or np.isnan(v)) else float(v) for v in r]
            for name, r in bench.items()}
        cached["_bench_dates"] = [str(pd.Timestamp(d).date()) for d in bt["date"]]

    series = {"model": [{"date": str(pd.Timestamp(d).date()), "value": float(v)}
                        for d, v in zip(bt["date"], (1 + bt["port"] - bt["cost"]).cumprod())]}
    stats = {"model": _stats(bt["port"] - bt["cost"])}
    for name, raw in cached["_bench"].items():
        r = pd.Series([np.nan if v is None else v for v in raw], index=bt["date"])
        curve = (1 + r.fillna(0)).cumprod()
        series[name] = [{"date": str(pd.Timestamp(d).date()), "value": float(v)}
                        for d, v in curve.items()]
        stats[name] = _stats(r.to_numpy())

    result = {"series": series, "stats": stats}
    cached[arm] = result
    _write_bench_cache(cached)
    return result


def _read_bench_cache():
    if not BENCH_CACHE.exists():
        return {}
    try:
        return json.loads(BENCH_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_bench_cache(data):
    BENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BENCH_CACHE.write_text(json.dumps(data))
