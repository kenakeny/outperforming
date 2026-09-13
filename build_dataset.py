"""
build_dataset.py  —  one self-contained script to GET and CREATE the dataset.

Downloads the US-ETF universe (yfinance + financedatabase), then engineers the
exact 81-feature design matrix + peer-relative label the model trains on.
No other project files are needed — send this one file.

WHAT IT DOES
  1. pull every US ETF from financedatabase, download 10y daily OHLCV via yfinance
  2. keep funds with >= 252 days of history, then keep only funds whose
     financedatabase `category` has >= 3 members (the peer group the label ranks
     within); drop 'Uncategorized'
  3. build 10 feature families (81 columns), all strictly trailing (data <= t)
  4. build the label: next-HORIZON-day return minus the leave-one-out category
     peer mean -> per-day terciles (0=under / 1=neutral / 2=over)

OUTPUTS (under ./data)
  data/raw/market_data.parquet   wide OHLCV panel (MultiIndex cols: field x ticker)
  data/raw/metadata.parquet      per-ticker name / category / exchange
  data/processed/features.parquet   long (date, ticker) x 81 features
  data/processed/labels.parquet     long (date, ticker): target, rel, fwd_ret

USAGE
  pip install yfinance financedatabase pandas numpy pyarrow
  python build_dataset.py                 # horizon defaults to 10 trading days
  python build_dataset.py --horizon 5     # 1-week label instead of 2-week
  python build_dataset.py --period 5y     # shorter history (faster)

Runtime: ~10-20 min, dominated by the yfinance download (~1,700 funds).
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  config                                                                      #
# --------------------------------------------------------------------------- #
EXCHANGES = ["PCX", "NMS", "NGM", "NYQ", "ASE"]   # NYSE Arca, Nasdaq, Nasdaq GM, NYSE, AMEX
MIN_DAYS = 252        # drop funds with < ~1y of history
MIN_GROUP = 3         # a category needs >= this many funds to be a peer group
PERIOD = "10y"
INTERVAL = "1d"

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


# --------------------------------------------------------------------------- #
#  1. download raw universe                                                    #
# --------------------------------------------------------------------------- #
def download_universe(period, interval):
    import yfinance as yf
    import financedatabase as fd

    etfs = fd.ETFs().data
    uni = etfs[etfs["exchange"].isin(EXCHANGES)].copy()
    uni.index = uni.index.astype(str).str.strip()
    uni = uni[~uni.index.duplicated()]
    tickers = sorted(uni.index.unique())
    print(f"{len(tickers)} ETFs across {uni['category'].nunique()} categories")

    out, batch = [], 300
    for i in range(0, len(tickers), batch):
        b = tickers[i:i + batch]
        df = yf.download(b, period=period, interval=interval, auto_adjust=True,
                         progress=False, group_by="column", threads=True)
        if df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):          # single-ticker batch
            df.columns = pd.MultiIndex.from_product([df.columns, b])
        out.append(df)
        print(f"  downloaded {min(i + batch, len(tickers)):>5}/{len(tickers)}")

    md = pd.concat(out, axis=1)
    md = md.loc[:, ~md.columns.duplicated()].sort_index()
    md.index.name = "date"
    return md, uni


def clean_and_save_raw(md, uni):
    """min-days filter, build metadata, persist raw parquets. Returns (md, meta)."""
    close = md.xs("Close", axis=1, level=0)
    keep = close.notna().sum() >= MIN_DAYS
    kept_cols = close.columns[keep]
    md = md.loc[:, md.columns.get_level_values(1).isin(kept_cols)]

    meta_cols = ["name", "category_group", "category", "family", "currency", "exchange"]
    meta = uni.reindex(kept_cols)[meta_cols].copy()
    meta.index.name = "ticker"
    meta["category"] = meta["category"].fillna("Uncategorized")
    meta["category_group"] = meta["category_group"].fillna("Uncategorized")

    RAW.mkdir(parents=True, exist_ok=True)
    md.to_parquet(RAW / "market_data.parquet")
    meta.to_parquet(RAW / "metadata.parquet")
    print(f"saved raw: {len(kept_cols)} funds with >= {MIN_DAYS} days -> {RAW}")
    return md, meta


def make_universe(md, meta):
    """Restrict to funds whose category has >= MIN_GROUP peers (not Uncategorized).
    Returns (fields, cat): fields = {Open/High/Low/Close/Volume -> wide frame}."""
    md.index = pd.to_datetime(md.index); md.index.name = "date"
    cat = meta["category"].reindex(md["Close"].columns)
    counts = cat.value_counts()
    keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
    kept = cat[cat.isin(keep)].index
    fields = {}
    for f in ["Open", "High", "Low", "Close", "Volume"]:
        w = md[f][kept]; w.columns.name = "ticker"; fields[f] = w
    print(f"modeling universe: {len(kept)} funds in {len(keep)} categories")
    return fields, cat[kept]


# --------------------------------------------------------------------------- #
#  shared helpers  (peer grouping is leave-one-out so a fund is never in its    #
#  own benchmark)                                                              #
# --------------------------------------------------------------------------- #
def stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s


def assemble(feat_dict):
    """dict[name] -> wide frame  ==>  long (date, ticker) design matrix."""
    return pd.DataFrame({k: stack(v) for k, v in feat_dict.items()})


def loo_peer_mean(x, cat):
    """Leave-one-out category-peer mean of wide frame x, per day."""
    s = x.T.groupby(cat).transform("sum").T
    n = x.notna().T.groupby(cat).transform("sum").T
    return (s - x.fillna(0)) / (n - x.notna().astype(int)).replace(0, np.nan)


def cat_pct_rank(x, cat):
    """Cross-sectional percentile rank (0..1) of x within category, per day."""
    return x.T.groupby(cat).rank(pct=True).T


def build_label(close, cat, horizon, skip=0):
    """Next-`horizon`-day peer-relative tercile. Returns long frame with
    target (0/1/2), rel (continuous peer-rel excess), fwd_ret."""
    fwd = close.shift(-horizon) / close.shift(-skip) - 1.0
    rel = fwd - loo_peer_mean(fwd, cat)
    fwd_l, rel_l = stack(fwd), stack(rel)
    target = (rel_l.dropna().groupby(level="date")
              .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                           labels=[0, 1, 2])).astype("int8"))
    return (pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1)
            .dropna(subset=["target"]))


# --------------------------------------------------------------------------- #
#  2. feature families  (10 families, 81 columns — order matters for dedup)     #
# --------------------------------------------------------------------------- #
def fam_01_voltail(fields, cat):
    close, high, low = fields["Close"], fields["High"], fields["Low"]
    r1 = close.pct_change(fill_method=None)
    vol_20d = r1.rolling(20).std()
    vol_cat_rank = cat_pct_rank(vol_20d, cat)
    hl = (high - low) / close
    return {
        "vol_20d": vol_20d,
        "vol_pctile_1y": vol_20d.rolling(252).rank(pct=True),
        "vol_of_vol_60d": vol_20d.rolling(60).std() / vol_20d.rolling(60).mean(),
        "vol_spike": r1.rolling(5).std() / r1.rolling(60).std(),
        "vol_cat_rank": vol_cat_rank,
        "vol_top_decile": (vol_cat_rank >= 0.9).astype(float).where(vol_cat_rank.notna()),
        "hl_spike": hl.rolling(5).mean() / hl.rolling(60).mean(),
    }


def fam_02_peermom(fields, cat):
    close = fields["Close"]
    fd = {}
    for n in (5, 20, 60):
        ret_n = close.pct_change(n, fill_method=None)
        fd[f"ret_{n}d_peerrel"] = ret_n - loo_peer_mean(ret_n, cat)
        fd[f"ret_{n}d_catrank"] = cat_pct_rank(ret_n, cat)
    r = close.shift(5) / close.shift(25) - 1.0
    fd["ret_20d_skip5_peerrel"] = r - loo_peer_mean(r, cat)
    up_frac = (close.pct_change(fill_method=None) > 0).rolling(20).mean()
    fd["mom_consistency_20d"] = cat_pct_rank(up_frac, cat)
    d1 = close.pct_change(fill_method=None)
    fd["streak_rel_5d"] = (d1 - loo_peer_mean(d1, cat)).rolling(5).sum()
    return fd


def fam_03_peerext(fields, cat):
    close = fields["Close"]
    roll_max, roll_min = close.rolling(252).max(), close.rolling(252).min()
    px_hi = close / roll_max - 1.0
    px_lo = close / roll_min - 1.0
    px_sma200 = close / close.rolling(200).mean() - 1.0
    px_sma50 = close / close.rolling(50).mean() - 1.0
    range_pos = (close - roll_min) / (roll_max - roll_min)
    return {
        "px_to_52w_high": px_hi,
        "px_to_52w_high_catrank": cat_pct_rank(px_hi, cat),
        "px_to_52w_low": px_lo,
        "px_to_52w_low_catrank": cat_pct_rank(px_lo, cat),
        "px_to_sma_200": px_sma200,
        "px_to_sma_200_catrank": cat_pct_rank(px_sma200, cat),
        "px_to_sma_200_peerrel": px_sma200 - loo_peer_mean(px_sma200, cat),
        "px_to_sma_50_catrank": cat_pct_rank(px_sma50, cat),
        "range_pos_52w": range_pos,
        "range_pos_52w_catrank": cat_pct_rank(range_pos, cat),
    }


def fam_04_liquidity(fields, cat):
    close, vol = fields["Close"], fields["Volume"]
    dv = close * vol
    r1 = close.pct_change(fill_method=None)
    dv20 = dv.rolling(20).mean()
    amihud = (r1.abs() / dv.replace(0, np.nan)).rolling(20).mean()
    z = (vol == 0) & close.notna()
    return {
        "dollar_vol_log": np.log1p(dv20),
        "dollar_vol_catrank": cat_pct_rank(dv20, cat),
        "amihud_20d": amihud,
        "amihud_catrank": cat_pct_rank(amihud, cat),
        "volume_trend": vol.rolling(20).mean() / vol.rolling(60).mean() - 1.0,
        "volume_spike_5d": vol.rolling(5).mean() / vol.rolling(60).mean() - 1.0,
        "zero_vol_frac_20d": z.rolling(20).sum() / close.notna().rolling(20).sum(),
        "turnover_vol_20d": dv.rolling(20).std() / dv20,
    }


def fam_05_betaidio(fields, cat):
    close = fields["Close"]
    r1 = close.pct_change(fill_method=None)
    m = loo_peer_mean(r1, cat)          # fund-specific category-index daily return
    ex1 = r1 - m                        # idiosyncratic (peer-excess) daily return
    beta = r1.rolling(60).cov(m) / m.rolling(60).var().replace(0, np.nan)
    idio_vol = ex1.rolling(20).std()
    return {
        "beta_60d": beta,
        "beta_catrank": cat_pct_rank(beta, cat),
        "corr_cat_60d": r1.rolling(60).corr(m),
        "resid_ret_5d": ex1.rolling(5).sum(),
        "resid_ret_20d": ex1.rolling(20).sum(),
        "idio_vol_20d": idio_vol,
        "idio_vol_catrank": cat_pct_rank(idio_vol, cat),
        "idio_frac_20d": idio_vol / r1.rolling(20).std().replace(0, np.nan),
        "tstat_rel_5d": ex1.rolling(5).mean() / (ex1.rolling(5).std().replace(0, np.nan) / np.sqrt(5)),
    }


def fam_06_catcontext(fields, cat):
    close = fields["Close"]
    r1 = close.pct_change(fill_method=None)
    r5 = close.pct_change(5, fill_method=None)
    r20 = close.pct_change(20, fill_method=None)
    v = r1.rolling(20).std()
    return {
        "cat_disp_5d": r5.T.groupby(cat).transform("std").T,
        "cat_disp_20d": r20.T.groupby(cat).transform("std").T,
        "cat_ret_5d": r5.T.groupby(cat).transform("mean").T,
        "cat_ret_20d": r20.T.groupby(cat).transform("mean").T,
        "cat_vol_20d": v.T.groupby(cat).transform("mean").T,
        "cat_n": close.notna().T.groupby(cat).transform("sum").T,
        "cat_breadth_5d": (r5 > 0).where(r5.notna()).T.groupby(cat).transform("mean").T,
        "disp_x_vol_rank": cat_pct_rank(v, cat) * r5.T.groupby(cat).transform("std").T,
    }


def fam_07_range(fields, cat):
    openp, high, low, close = (fields[k] for k in ("Open", "High", "Low", "Close"))
    low_s, open_s, close_s = low.replace(0, np.nan), openp.replace(0, np.nan), close.replace(0, np.nan)
    log_hl = np.log((high / low_s).clip(lower=1e-12))
    log_co = np.log((close / open_s).clip(lower=1e-12))
    pk = (log_hl ** 2) / (4 * np.log(2))
    parkinson = np.sqrt(pk.rolling(20).mean())
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    clv = (2 * close - high - low) / (high - low).replace(0, np.nan)
    gap = openp / close.shift(1).replace(0, np.nan) - 1.0
    rel_range = (high - low) / close_s
    ccvol = close.pct_change(fill_method=None).rolling(20).std()
    return {
        "parkinson_20d": parkinson,
        "gk_20d": np.sqrt(gk.clip(lower=0).rolling(20).mean()),
        "parkinson_catrank": cat_pct_rank(parkinson, cat),
        "clv_20d": clv.rolling(20).mean(),
        "gap_20d": gap.rolling(20).mean(),
        "gap_abs_20d": gap.abs().rolling(20).mean(),
        "range_ratio_5_60": rel_range.rolling(5).mean() / rel_range.rolling(60).mean().replace(0, np.nan),
        "pk_to_ccvol": parkinson / ccvol.replace(0, np.nan),
    }


def fam_11_downside(fields, cat):
    close, high, low = fields["Close"], fields["High"], fields["Low"]
    r1 = close.pct_change(fill_method=None)
    dd_60d = close / close.rolling(60).max() - 1.0
    dd_252d = close / close.rolling(252).max() - 1.0

    # days since the trailing 60d high (0 = at high today)
    W = 60
    vals = close.ffill().to_numpy()
    days_since = np.full(vals.shape, np.nan)
    if len(vals) >= W:
        win = np.lib.stride_tricks.sliding_window_view(vals, W, axis=0)   # (T-W+1, N, W)
        with np.errstate(invalid="ignore"):
            am = np.nanargmax(np.where(np.isnan(win), -np.inf, win), axis=-1)
        days_since[W - 1:] = (W - 1) - am
    days_since_high = pd.DataFrame(days_since, index=close.index,
                                   columns=close.columns).where(close.notna())

    semivol = r1.where(r1 < 0).rolling(20, min_periods=5).std()
    return {
        "dd_60d": dd_60d,
        "dd_252d": dd_252d,
        "days_since_high_60d": days_since_high,
        "downside_share_20d": semivol / r1.rolling(20).std(),
        "sortino_20d": r1.rolling(20).mean() / semivol,
        "skew_60d": r1.rolling(60).skew(),
        "kurt_60d": r1.rolling(60).kurt(),
        "dd_catrank": cat_pct_rank(dd_60d, cat),
    }


def fam_12_temporal(fields, cat):
    close = fields["Close"]
    r1 = close.pct_change(fill_method=None)
    sharpe_mom_20 = close.pct_change(20, fill_method=None) / (r1.rolling(20).std() * np.sqrt(20))
    ret_20d = close.pct_change(20, fill_method=None)
    logp = np.log(close)
    tgrid = pd.DataFrame(np.tile(np.arange(len(close), dtype=float)[:, None], (1, close.shape[1])),
                         index=close.index, columns=close.columns)
    tr_corr = logp.rolling(20).corr(tgrid)
    return {
        "autocorr1_60d": r1.rolling(60).corr(r1.shift(1)),
        "sharpe_mom_20d": sharpe_mom_20,
        "sharpe_mom_60d": close.pct_change(60, fill_method=None) / (r1.rolling(60).std() * np.sqrt(60)),
        "mom_accel_20d": ret_20d - ret_20d.shift(20),
        "updays_frac_20d": (r1 > 0).astype(float).where(r1.notna()).rolling(20, min_periods=15).mean(),
        "trend_str_20d": tr_corr.pow(2) * np.sign(tr_corr),
        "sharpe_mom_catrank": cat_pct_rank(sharpe_mom_20, cat),
    }


def fam_15_overnight(fields, cat):
    o, c = fields["Open"], fields["Close"]
    gap = o / c.shift(1) - 1.0                       # overnight (close->open)
    intra = c / o - 1.0                              # intraday (open->close)
    on_ret = gap.rolling(20).sum()
    id_ret = intra.rolling(20).sum()
    denom = gap.abs().rolling(20).sum() + intra.abs().rolling(20).sum()
    return {
        "on_ret_20d": on_ret,
        "id_ret_20d": id_ret,
        "on_share_20d": gap.abs().rolling(20).sum() / denom.replace(0, np.nan),
        "on_minus_id_20d": on_ret - id_ret,
        "on_vol_ratio_20d": gap.rolling(20).std() / intra.rolling(20).std().replace(0, np.nan),
        "on_ret_catrank": cat_pct_rank(on_ret, cat),
        "id_ret_catrank": cat_pct_rank(id_ret, cat),
    }


FAMILIES = [                       # order matters: duplicate columns keep the first
    fam_01_voltail, fam_02_peermom, fam_03_peerext, fam_04_liquidity,
    fam_05_betaidio, fam_06_catcontext, fam_07_range, fam_11_downside,
    fam_12_temporal, fam_15_overnight,
]


def build_features(fields, cat):
    frames = []
    for fam in FAMILIES:
        X = assemble(fam(fields, cat))
        print(f"  {fam.__name__:20s} -> {X.shape[1]} features")
        frames.append(X)
    X = pd.concat(frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated(keep="first")].astype("float32")
    X = X.mask(np.isinf(X))         # zero-denominator ratios -> NaN (xgboost-safe)
    return X


# --------------------------------------------------------------------------- #
#  main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10, help="label horizon in trading days")
    ap.add_argument("--period", default=PERIOD, help="yfinance history, e.g. 10y / 5y")
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse existing data/raw/*.parquet instead of downloading")
    args = ap.parse_args()

    if args.skip_download and (RAW / "market_data.parquet").exists():
        print("reusing existing raw data")
        md = pd.read_parquet(RAW / "market_data.parquet")
        meta = pd.read_parquet(RAW / "metadata.parquet")
    else:
        md, uni = download_universe(args.period, INTERVAL)
        md, meta = clean_and_save_raw(md, uni)

    fields, cat = make_universe(md, meta)

    print("building features ...")
    X = build_features(fields, cat)
    print(f"design matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    print(f"building label (horizon={args.horizon} trading days) ...")
    lbl = build_label(fields["Close"], cat, args.horizon)

    PROC.mkdir(parents=True, exist_ok=True)
    X.to_parquet(PROC / "features.parquet")
    lbl.to_parquet(PROC / "labels.parquet")
    print(f"\nDONE")
    print(f"  {PROC / 'features.parquet'}   {X.shape[0]:,} x {X.shape[1]}")
    print(f"  {PROC / 'labels.parquet'}     {lbl.shape[0]:,} rows "
          f"(target balance: {np.bincount(lbl['target']).tolist()})")


if __name__ == "__main__":
    main()
