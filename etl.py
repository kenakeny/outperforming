"""
etl.py -- the reproducible ETL pipeline behind dataset_v3.

Until now the v3 design matrix existed only as executed cells in
`notebooks/03_features.ipynb`: `data/processed/dataset_v3.parquet` (the file every
model in `train_models.py` trains on) had no script that could rebuild it. This
module is that script -- the same feature definitions, lifted out of the notebook
into importable, testable stage functions.

STAGES (each writes one parquet, each skippable if its output is fresh)
  extract   yfinance OHLCV + financedatabase metadata -> data/raw/market_data.parquet
  features  44 trailing price/volume/cross-sectional features -> features_v3.parquet
  sr        11 support/resistance features             -> sr_features.parquet
  labels    peer-relative forward-return terciles      -> labels.parquet
  assemble  join features + labels                     -> dataset_v3.parquet

USAGE
  python etl.py --stages all
  python etl.py --stages features,sr,assemble      # skip the slow download
  python etl.py --stages all --force               # ignore freshness checks
  python etl.py --status                           # what exists, how stale

Every feature here is strictly trailing (uses data through t only); `tests/` asserts
that by truncation, and the label is the only column that looks forward.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MANIFEST = PROC / "etl_manifest.json"

STAGE_ORDER = ["extract", "features", "sr", "labels", "assemble"]

STAGE_OUTPUTS = {
    "extract": RAW / "market_data.parquet",
    "features": PROC / "features_v3.parquet",
    "sr": PROC / "sr_features.parquet",
    "labels": PROC / "labels.parquet",
    "assemble": PROC / "dataset_v3.parquet",
}

# the 44 feature columns of dataset_v3, in order. Anything that trains on the
# dataset resolves its feature list from here rather than "every column that
# isn't the label" -- that guesswork is how a label column becomes a feature.
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "ema20_dist", "ema50_dist", "macd", "macd_signal", "macd_hist", "adx_14",
    "rsi_14", "willr_14", "bb_pctb", "stoch_k", "stoch_d",
    "vol_20d", "vol_zscore", "bb_bandwidth", "atr_14", "beta_60d",
    "obv", "mfi_14", "cmf_20", "dollar_vol_20d", "rel_volume_20d",
    "cs_rank_ret_5d", "cs_rank_ret_20d", "cs_rank_rsi", "cs_rank_volatility",
    "cs_rank_volume", "peer_rank_20d",
    "d_rsi_5d", "d_peer_rank_5d", "d_vol_5d", "d_macd_hist_5d", "d_volume_pct_5d",
    "cat_ret_5d", "cat_disp_5d", "beat_rate_20d",
    "mkt_ret_1d", "mkt_vol_20d", "mkt_vol_zscore", "breadth_50d",
]
LABEL_COLS = ["target", "fwd_ret", "rel"]

# market-wide features are one value per date, broadcast to every ticker on join
REGIME_COLS = ["mkt_ret_1d", "mkt_vol_20d", "mkt_vol_zscore", "breadth_50d"]

SR_WINDOWS = [20, 60]
SR_TOL = 0.02


def load_config(path=None):
    return yaml.safe_load(open(path or ROOT / "config.yaml", encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  panel assembly                                                             #
# --------------------------------------------------------------------------- #

def clean_panel(market_data, metadata, min_group):
    """Restrict the OHLCV panel to funds that can carry a peer-relative label.

    Drops leveraged/inverse funds and bad-tick funds (their returns aren't
    comparable to their peers), then drops any category with fewer than
    `min_group` members -- a tercile over two funds isn't a tercile.

    Returns (close, high, low, volume, cat) with aligned index/columns.
    """
    close = market_data["Close"].copy()
    close.index.name, close.columns.name = "date", "ticker"

    flags = pd.Series(False, index=metadata.index)
    for col in ("is_leveraged", "bad_ticks"):
        if col in metadata.columns:
            flags |= metadata[col].fillna(False).astype(bool)
    close = close.drop(columns=[t for t in metadata.index[flags] if t in close.columns])

    cat = metadata["category"].reindex(close.columns)
    sizes = cat.value_counts()
    keep_cats = sizes[(sizes >= min_group) & (sizes.index != "Uncategorized")].index
    keep = cat[cat.isin(keep_cats)].index
    close, cat = close[keep], cat[keep]

    reidx = dict(index=close.index, columns=close.columns)
    return (close,
            market_data["High"].reindex(**reidx),
            market_data["Low"].reindex(**reidx),
            market_data["Volume"].reindex(**reidx),
            cat)


# --------------------------------------------------------------------------- #
#  stage: features                                                            #
# --------------------------------------------------------------------------- #

def build_features(close, high, low, volume, cat):
    """The 44 v3 features as wide (date x ticker) frames, plus the 4 market-wide
    regime series. Ported verbatim from notebooks/03_features.ipynb.

    Returns (dict[name] -> wide frame, DataFrame of regime series indexed by date).
    Every frame is a function of data through t -- no shift(-k) anywhere.
    """
    f = {}
    for w in (1, 5, 10, 20, 60):
        f[f"ret_{w}d"] = close.pct_change(w, fill_method=None)
    ret_1d, ret_5d, ret_20d = f["ret_1d"], f["ret_5d"], f["ret_20d"]

    # --- trend ---
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    f["ema20_dist"] = close / ema_20 - 1.0
    f["ema50_dist"] = close / ema_50 - 1.0
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_signal"] = macd, macd_signal
    macd_hist = macd - macd_signal
    f["macd_hist"] = macd_hist

    # ADX/ATR -- Wilder smoothing is ewm(alpha=1/period, adjust=False)
    p = 14
    prev_close = close.shift(1)
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr_14 = tr.ewm(alpha=1 / p, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / p, adjust=False).mean() / atr_14
    minus_di = 100 * minus_dm.ewm(alpha=1 / p, adjust=False).mean() / atr_14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    f["adx_14"], f["atr_14"] = dx.ewm(alpha=1 / p, adjust=False).mean(), atr_14

    # --- mean reversion ---
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_14 = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    f["rsi_14"] = rsi_14

    hh, ll = high.rolling(14).max(), low.rolling(14).min()
    hl_range = (hh - ll).replace(0, np.nan)   # flat 14d window -> NaN, not inf
    f["willr_14"] = (-100 * (hh - close) / hl_range).clip(-100, 0)
    stoch_k = (100 * (close - ll) / hl_range).clip(0, 100)
    f["stoch_k"], f["stoch_d"] = stoch_k, stoch_k.rolling(3).mean()

    bb_mid, bb_std = close.rolling(20).mean(), close.rolling(20).std()
    bb_upper, bb_lower = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    f["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower)
    f["bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid

    # --- volatility / beta ---
    vol_20d = ret_1d.rolling(20).std()
    f["vol_20d"] = vol_20d
    f["vol_zscore"] = ((vol_20d - vol_20d.rolling(252, min_periods=60).mean())
                       / vol_20d.rolling(252, min_periods=60).std())

    cat_mean_ret_1d = ret_1d.T.groupby(cat).transform("mean").T
    w = 60
    roll_x = ret_1d.rolling(w).mean()
    roll_m = cat_mean_ret_1d.rolling(w).mean()
    cov_xm = (ret_1d * cat_mean_ret_1d).rolling(w).mean() - roll_x * roll_m
    var_m = (cat_mean_ret_1d ** 2).rolling(w).mean() - roll_m ** 2
    f["beta_60d"] = cov_xm / var_m

    # --- volume ---
    f["obv"] = (np.sign(ret_1d).fillna(0.0) * volume.fillna(0.0)).cumsum()
    f["dollar_vol_20d"] = (close * volume).rolling(20).mean()

    typical = (high + low + close) / 3
    money_flow = typical * volume
    tp_change = typical.diff()
    pos = money_flow.where(tp_change > 0, 0.0).rolling(14).sum()
    neg = money_flow.where(tp_change < 0, 0.0).rolling(14).sum()
    f["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl.where(hl > 0)).clip(-1, 1)
    mfm = mfm.mask((hl == 0) & close.notna(), 0.0)  # flat bar = zero flow, don't poison the window
    f["cmf_20"] = (mfm * volume).rolling(20).sum() / volume.rolling(20).sum()
    f["rel_volume_20d"] = volume / volume.rolling(20).mean() - 1.0

    # --- cross-sectional ranks ---
    cs_rank_volume = volume.rank(axis=1, pct=True)
    f["cs_rank_ret_5d"] = ret_5d.rank(axis=1, pct=True)
    f["cs_rank_ret_20d"] = ret_20d.rank(axis=1, pct=True)
    f["cs_rank_rsi"] = rsi_14.rank(axis=1, pct=True)
    f["cs_rank_volatility"] = vol_20d.rank(axis=1, pct=True)
    f["cs_rank_volume"] = cs_rank_volume
    peer_rank_20d = ret_20d.T.groupby(cat).rank(pct=True).T
    f["peer_rank_20d"] = peer_rank_20d

    # --- deltas ---
    f["d_rsi_5d"] = rsi_14 - rsi_14.shift(5)
    f["d_peer_rank_5d"] = peer_rank_20d - peer_rank_20d.shift(5)
    f["d_vol_5d"] = vol_20d - vol_20d.shift(5)
    f["d_macd_hist_5d"] = macd_hist - macd_hist.shift(5)
    f["d_volume_pct_5d"] = cs_rank_volume - cs_rank_volume.shift(5)

    # --- category context ---
    cat_ret_20d = ret_20d.T.groupby(cat).transform("mean").T
    f["cat_ret_5d"] = ret_5d.T.groupby(cat).transform("mean").T
    f["cat_disp_5d"] = ret_5d.T.groupby(cat).transform("std").T
    f["beat_rate_20d"] = (ret_20d > cat_ret_20d).T.groupby(cat).transform("mean").T

    # --- market regime (one series per date) ---
    mkt_ret_1d = ret_1d.mean(axis=1)
    mkt_vol_20d = mkt_ret_1d.rolling(20).std()
    regime = pd.DataFrame({
        "mkt_ret_1d": mkt_ret_1d,
        "mkt_vol_20d": mkt_vol_20d,
        "mkt_vol_zscore": ((mkt_vol_20d - mkt_vol_20d.rolling(252, min_periods=60).mean())
                           / mkt_vol_20d.rolling(252, min_periods=60).std()),
        "breadth_50d": (close > ema_50).sum(axis=1) / close.notna().sum(axis=1),
    }).astype(np.float32)
    regime.index.name = "date"

    return f, regime


def features_to_long(frames, regime):
    """Wide (date x ticker) frames -> a single long (date, ticker) x 44 frame."""
    def stack(df):
        s = df.stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        return s.astype(np.float32)

    per_ticker = [c for c in FEATURE_COLS if c not in REGIME_COLS]
    long = pd.concat({name: stack(frames[name]) for name in per_ticker}, axis=1)
    long = long.join(regime, on="date")
    return long[FEATURE_COLS].sort_index()


# --------------------------------------------------------------------------- #
#  stage: support / resistance                                                #
# --------------------------------------------------------------------------- #

def build_sr_features(close, high, low, windows=SR_WINDOWS, tol=SR_TOL):
    """Swing-based support/resistance, as wide frames.

    Resistance is the rolling max of High and support the rolling min of Low over
    the *prior* w days -- `.shift(1)` before `.rolling()`, so a value at t sees
    only bars through t-1 plus today's close. Same construction as the standalone
    build_sr_features.py, moved here so the pipeline owns one copy.
    """
    out = {}
    for w in windows:
        res = high.shift(1).rolling(w).max()
        sup = low.shift(1).rolling(w).min()
        span = (res - sup).replace(0, np.nan)
        out[f"dist_res_{w}"] = (res - close) / close
        out[f"dist_sup_{w}"] = (close - sup) / close
        out[f"range_pos_{w}"] = (close - sup) / span

        if w == 20:   # the actionable timescale gets the richer set
            out["sr_width_20"] = (res - sup) / close
            out["res_touch_20"] = (high >= res * (1 - tol)).astype(float).shift(1).rolling(w).sum()
            out["sup_touch_20"] = (low <= sup * (1 + tol)).astype(float).shift(1).rolling(w).sum()
            out["broke_res_20"] = (close > res).astype(float)
            out["broke_sup_20"] = (close < sup).astype(float)
    return out


def sr_to_long(frames):
    long = pd.concat({k: v.stack() for k, v in frames.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    return long.replace([np.inf, -np.inf], np.nan).dropna(how="all").sort_index()


# --------------------------------------------------------------------------- #
#  stage: labels                                                              #
# --------------------------------------------------------------------------- #

def build_labels(close, cat, horizon, min_group):
    """Peer-relative forward-return terciles -- the only forward-looking column.

    fwd_ret is the h-day forward return; `rel` subtracts the leave-one-out category
    mean (a fund is not its own benchmark); `target` is the within-(date, category)
    tercile of `rel`: 0 = underperform, 1 = neutral, 2 = outperform. Days where a
    category has < min_group members get NaN rather than a meaningless tercile.
    """
    fwd = close.shift(-horizon) / close - 1.0

    # leave-one-out category mean: (sum - self) / (n - 1)
    grp = fwd.T.groupby(cat)
    cat_sum = grp.transform("sum").T
    cat_n = grp.transform("count").T
    loo = (cat_sum - fwd) / (cat_n - 1).where(cat_n > 1)
    rel = fwd - loo

    long = pd.concat({"fwd_ret": fwd.stack(future_stack=True),
                      "rel": rel.stack(future_stack=True)}, axis=1)
    long.index.names = ["date", "ticker"]
    long["cat"] = long.index.get_level_values("ticker").map(cat)

    g = long.dropna(subset=["rel"]).groupby(["date", "cat"])["rel"]
    pct, size = g.rank(pct=True), g.transform("size")
    target = pd.Series(np.select([pct <= 1 / 3, pct <= 2 / 3], [0.0, 1.0], default=2.0),
                       index=pct.index)
    long["target"] = target.where(size >= min_group)
    return long[LABEL_COLS].sort_index()


# --------------------------------------------------------------------------- #
#  stage runners                                                              #
# --------------------------------------------------------------------------- #

def stage_extract(cfg, force=False):
    """Download the raw OHLCV panel. Delegates to build_dataset.py, which already
    owns the yfinance/financedatabase universe logic."""
    import build_dataset
    md, uni = build_dataset.download_universe(cfg["data"]["period"], cfg["data"]["interval"])
    RAW.mkdir(parents=True, exist_ok=True)
    md.to_parquet(RAW / "market_data.parquet")
    uni.to_parquet(RAW / "metadata.parquet")
    return {"rows": len(md), "tickers": md["Close"].shape[1]}


def _load_panel(cfg):
    md = pd.read_parquet(RAW / "market_data.parquet").sort_index()
    meta = pd.read_parquet(RAW / "metadata.parquet")
    return clean_panel(md, meta, cfg["labels"]["min_group"])


def stage_features(cfg, force=False):
    close, high, low, volume, cat = _load_panel(cfg)
    frames, regime = build_features(close, high, low, volume, cat)
    long = features_to_long(frames, regime)
    PROC.mkdir(parents=True, exist_ok=True)
    long.to_parquet(PROC / "features_v3.parquet")
    return {"rows": len(long), "cols": len(long.columns)}


def stage_sr(cfg, force=False):
    close, high, low, _, _ = _load_panel(cfg)
    long = sr_to_long(build_sr_features(close, high, low))
    long.to_parquet(PROC / "sr_features.parquet")
    return {"rows": len(long), "cols": len(long.columns)}


def stage_labels(cfg, force=False):
    close, _, _, _, cat = _load_panel(cfg)
    long = build_labels(close, cat, cfg["labels"]["horizon"], cfg["labels"]["min_group"])
    long.to_parquet(PROC / "labels.parquet")
    return {"rows": len(long), "labelled": int(long["target"].notna().sum())}


def stage_assemble(cfg, force=False):
    feats = pd.read_parquet(PROC / "features_v3.parquet")
    labels = pd.read_parquet(PROC / "labels.parquet")
    d = feats.join(labels, how="inner").dropna(subset=["target"])
    d["target"] = d["target"].astype(np.int8)
    d.to_parquet(PROC / "dataset_v3.parquet")
    return {"rows": len(d), "features": len(FEATURE_COLS),
            "class_balance": d["target"].value_counts(normalize=True).round(4).to_dict()}


STAGES = {"extract": stage_extract, "features": stage_features, "sr": stage_sr,
          "labels": stage_labels, "assemble": stage_assemble}


# --------------------------------------------------------------------------- #
#  orchestration                                                              #
# --------------------------------------------------------------------------- #

def read_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def write_manifest(m):
    PROC.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, default=str))


def is_fresh(stage):
    """A stage is fresh if its output exists and is newer than every upstream
    stage's output. Cheap mtime check -- enough to skip re-running the 20-minute
    download when only the feature code changed."""
    out = STAGE_OUTPUTS[stage]
    if not out.exists():
        return False
    upstream = STAGE_ORDER[:STAGE_ORDER.index(stage)]
    return all(not STAGE_OUTPUTS[u].exists() or STAGE_OUTPUTS[u].stat().st_mtime <= out.stat().st_mtime
               for u in upstream)


def run(stages, cfg=None, force=False):
    cfg = cfg or load_config()
    manifest = read_manifest()
    for stage in stages:
        if not force and is_fresh(stage):
            print(f"[skip]  {stage:9} (output fresh -- use --force to rebuild)")
            continue
        print(f"[run]   {stage:9} ...", flush=True)
        t0 = time.time()
        info = STAGES[stage](cfg, force=force)
        elapsed = time.time() - t0
        manifest[stage] = {"finished": pd.Timestamp.now().isoformat(),
                           "seconds": round(elapsed, 1),
                           "output": str(STAGE_OUTPUTS[stage].relative_to(ROOT)), **info}
        print(f"[done]  {stage:9} {elapsed:6.1f}s  {info}")
        write_manifest(manifest)
    return manifest


def parse_stages(arg):
    if arg == "all":
        return list(STAGE_ORDER)
    stages = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}. valid: {STAGE_ORDER}")
    return sorted(stages, key=STAGE_ORDER.index)


def print_status():
    manifest = read_manifest()
    print(f"{'stage':10} {'output':40} {'exists':>7} {'fresh':>6}  last run")
    for stage in STAGE_ORDER:
        out = STAGE_OUTPUTS[stage]
        rec = manifest.get(stage, {})
        print(f"{stage:10} {str(out.relative_to(ROOT)):40} "
              f"{str(out.exists()):>7} {str(is_fresh(stage)):>6}  {rec.get('finished', '-')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all", help="'all' or comma-separated: " + ",".join(STAGE_ORDER))
    ap.add_argument("--force", action="store_true", help="rebuild even if the output looks fresh")
    ap.add_argument("--status", action="store_true", help="show stage status and exit")
    args = ap.parse_args()

    if args.status:
        print_status()
        return
    run(parse_stages(args.stages), force=args.force)


if __name__ == "__main__":
    main()
