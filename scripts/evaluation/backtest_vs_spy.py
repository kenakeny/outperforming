"""
Compare a saved model with SPY.

    python -m scripts.evaluation.backtest_vs_spy models/catboost_sr.cbm
    python -m scripts.evaluation.backtest_vs_spy models/xgboost_sr.ubj --year 2025
    python -m scripts.evaluation.backtest_vs_spy models/logreg_sr.joblib --plot

Loads the model, reads the feature names it was trained with, scores every fund
in the test year, holds the top N% for `horizon` days, and compares the result to
buying SPY over the identical windows.

Reports raw return AND beta/alpha, because a strategy can beat SPY by simply
taking more market risk -- alpha is what says it did something SPY didn't.
"""
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd

import core

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]

SPREADS = [(1e9, 1.0), (1e8, 3.0), (1e7, 10.0), (1e6, 30.0), (0.0, 80.0)]


def load_model(path):
    """Load by file extension and pull out the feature names it was fitted on."""
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"no such model: {path}")
    ext = path.suffix.lower()

    if ext == ".cbm":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(str(path))
        return m, list(m.feature_names_)
    if ext in (".ubj", ".json", ".bin"):
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        return m, list(m.get_booster().feature_names)
    if ext in (".joblib", ".pkl"):
        import joblib
        m = joblib.load(path)
        names = getattr(m, "feature_names_in_", None)
        if names is None:                      # sklearn Pipeline: ask the first step
            names = m[0].feature_names_in_
        return m, list(names)
    raise SystemExit(f"unsupported model type '{ext}' (.cbm/.ubj/.json/.joblib)")


def load_panel(feature_names, horizon, min_adv):
    """Whichever stored panel has the columns this model needs, plus a freshly
    computed `horizon`-day forward return (the panel's own label may be a
    different horizon, and silently reusing it is how horizons drift)."""
    close, high, low, volume, _ = core.etl._load_panel(core.etl.load_config())

    candidates = [core.etl.PROC / "dataset_v3.parquet"]
    d = None
    for p in candidates:
        if not p.exists():
            continue
        panel = pd.read_parquet(p)
        sr = core.etl.PROC / "sr_features.parquet"
        if sr.exists():
            panel = panel.join(pd.read_parquet(sr), how="left")
        news = core.etl.PROC / "news_features.parquet"
        if news.exists():
            nf = pd.read_parquet(news)
            panel = panel.join(nf, how="left")
            panel[nf.columns] = panel[nf.columns].fillna(0.0)
        if not set(feature_names) - set(panel.columns):
            d = panel
            break
    if d is None:                              # fall back to the scale-free set
        d, _ = core.panel(horizon)
        missing = set(feature_names) - set(d.columns)
        if missing:
            raise SystemExit(f"no panel has {len(missing)} feature(s) this model "
                             f"needs, e.g. {sorted(missing)[:6]}")

    fwd = (close.shift(-horizon) / close - 1.0).stack(future_stack=True)
    dv = (close * volume).rolling(20).mean().stack(future_stack=True)
    fwd.index.names = dv.index.names = ["date", "ticker"]
    d = d.drop(columns=[c for c in ("fwd_ret", "adv") if c in d.columns])
    d = d.join(pd.concat({"fwd_ret": fwd, "adv": dv}, axis=1), how="inner")
    d = d.dropna(subset=["fwd_ret", "adv"])
    d = d[d["fwd_ret"] != 0]
    return d[d["adv"] >= min_adv] if min_adv else d


def spy_forward(dates, horizon, retries=3):
    """SPY's `horizon`-day forward return at each rebalance date. yfinance fails
    intermittently, and an empty frame here used to surface as an opaque
    IndexError deep in the indexing code -- so fail loudly and retry."""
    import time

    import yfinance as yf
    for attempt in range(retries):
        s = yf.download("SPY", start="2015-01-01", auto_adjust=True, progress=False)
        if not s.empty:
            s = s["Close"]
            s = (s.iloc[:, 0] if hasattr(s, "columns") else s).dropna()
            if len(s) > horizon:
                f = s.shift(-horizon) / s - 1.0
                return np.array([f.iloc[min(s.index.searchsorted(pd.Timestamp(d)),
                                            len(s) - 1)] for d in dates], dtype=float)
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise SystemExit("could not download SPY from yfinance after "
                     f"{retries} attempts -- check the network and retry")


def summarize(r, py, label):
    sd = r.std()
    dd = (np.cumsum(r) - np.maximum.accumulate(np.cumsum(r))).min()
    return {"series": label, "ann": r.mean() * py, "vol": sd * np.sqrt(py),
            "sharpe": r.mean() * py / (sd * np.sqrt(py)) if sd else np.nan,
            "max_dd": dd}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="path to a saved model (.cbm/.ubj/.json/.joblib)")
    p.add_argument("--year", type=int, default=2026, help="test year")
    p.add_argument("--horizon", type=int, default=20, help="holding period, trading days")
    p.add_argument("--top", type=float, default=0.10, help="fraction of universe held")
    p.add_argument("--min-adv", type=float, default=50e6, help="liquidity screen")
    p.add_argument("--weight", default="score", choices=["equal", "score"])
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()

    model, feats = load_model(a.model)
    print(f"model   : {a.model}  ({len(feats)} features)")

    d = load_panel(feats, a.horizon, a.min_adv)
    d = d[d.index.get_level_values("date").year == a.year]
    if d.empty:
        raise SystemExit(f"no rows for {a.year}")
    d = d.dropna(subset=feats, how="all")

    proba = model.predict_proba(d[feats])
    sc = d.reset_index()[["date", "ticker", "fwd_ret", "adv"]]
    sc["score"] = proba[:, 2] - proba[:, 0]
    sc["vol_20d"] = d["vol_20d"].to_numpy() if "vol_20d" in d.columns else np.nan

    bps = sc["adv"].map(lambda x: next(b for f, b in SPREADS if x >= f)).mean()
    print(f"universe: {sc.ticker.nunique()} funds in {a.year}, "
          f"assumed spread {bps:.1f} bps")


    bt = core.backtest(sc, a.horizon, a.top, cost_bps=bps, weight=a.weight)
    if bt.empty:
        raise SystemExit("no rebalances -- try a smaller --horizon")

    spy = spy_forward(bt["date"], a.horizon)
    ok = ~np.isnan(spy)
    bt, spy = bt[ok].reset_index(drop=True), spy[ok]
    port = (bt["port"] - bt["cost"]).to_numpy()
    py = 252 / a.horizon

    rows = [summarize(port, py, "strategy (net)"),
            summarize(spy, py, "SPY"),
            summarize(bt["bench"].to_numpy(), py, "equal-weight universe")]
    out = pd.DataFrame(rows).set_index("series")
    out["ann"] = out["ann"].map("{:+.2%}".format)
    out["vol"] = out["vol"].map("{:.2%}".format)
    out["max_dd"] = out["max_dd"].map("{:.2%}".format)
    out["sharpe"] = out["sharpe"].round(2)

    ex = port - spy
    beta, alpha = np.polyfit(spy, port, 1)
    print(f"\nrebalances: {len(bt)}   turnover/rebal: {bt['turn'].mean():.2f}   "
          f"cost/rebal: {bt['cost'].mean()*1e4:.1f} bps\n")
    print(out.to_string())
    print(f"\nvs SPY:  excess {ex.mean()*py:+.2%}/yr   "
          f"IR {ex.mean()*py/(ex.std()*np.sqrt(py)):+.2f}   "
          f"hit {(ex > 0).mean():.1%}")
    print(f"         beta {beta:+.2f}   alpha {alpha*py:+.2%}/yr   "
          f"t={ex.mean()/(ex.std()/np.sqrt(len(ex))):+.2f}")
    if beta > 1.05 and ex.mean() > 0:
        print("         NOTE: beta > 1 -- some of that excess is just more market risk")

    # A model trained on a WITHIN-CATEGORY label scores "best bond fund" and "best
    # tech fund" alike, so ranking the whole universe by it holds the best fund in
    # each sleeve regardless of whether the sleeve is worth owning. The symptom is
    # picks that underperform the very universe they were drawn from.
    bench_ann = bt["bench"].mean() * py
    if port.mean() * py < bench_ann - 0.02:
        print(f"\n  NOTE: the strategy ({port.mean()*py:+.1%}) underperforms its own "
              f"equal-weight universe ({bench_ann:+.1%}).")
        print("  A model trained on a WITHIN-CATEGORY label produces scores that are only")
        print("  comparable to a fund's own peers -- ranking every fund by them picks the")
        print("  best fund in each sleeve, not the best sleeve. For a cross-sectional")
        print("  portfolio use a model trained on a universe-wide label")
        print("  (core.walkforward, or the experiments/ sweep).")
        if (meta_p := core.etl.RAW / "metadata.parquet").exists():
            cats = pd.read_parquet(meta_p)["category"]
            days = np.array(sorted(sc["date"].unique()))[::a.horizon]
            n = max(int(len(sc[sc["date"] == days[0]]) * a.top), 3)
            picks = pd.concat([sc[sc["date"] == d0].nlargest(n, "score") for d0 in days])
            pk = picks["ticker"].map(cats).value_counts(normalize=True)
            un = sc.groupby("ticker")["ticker"].first().map(cats).value_counts(normalize=True)
            tilt = (pk - un.reindex(pk.index).fillna(0)).sort_values(ascending=False)
            print(f"  biggest category overweights: "
                  + ", ".join(f"{k} {v:+.0%}" for k, v in tilt.head(3).items()))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.5))
        dt = pd.to_datetime(bt["date"])
        ax.plot(dt, np.cumsum(port) * 100, lw=2, color="#2a78d6",
                label=f"strategy ({port.mean()*py:+.1%}/yr)")
        ax.plot(dt, np.cumsum(spy) * 100, lw=2.5, color="#111",
                label=f"SPY ({spy.mean()*py:+.1%}/yr)")
        ax.plot(dt, np.cumsum(bt["bench"]) * 100, lw=1.6, ls="--", color="0.55",
                label="equal-weight universe")
        ax.axhline(0, color="#111", lw=1)
        ax.set_title(f"{pathlib.Path(a.model).name} vs SPY — {a.year}, "
                     f"top {a.top:.0%}, {a.horizon}d holds, net of costs")
        ax.set_ylabel("cumulative %")
        ax.legend()
        ax.grid(alpha=.3)
        outp = ROOT / "reports" / f"vs_spy_{pathlib.Path(a.model).stem}_{a.year}.png"
        outp.parent.mkdir(exist_ok=True)
        fig.savefig(outp, dpi=120, bbox_inches="tight")
        print(f"\nplot -> {outp}")


if __name__ == "__main__":
    main()
