"""
serve/csv_input.py -- turn a user-uploaded OHLCV CSV into the same feature panel
etl.py builds, so it can be scored by any registered model.

Reuses etl.build_features / etl.build_sr_features directly rather than
reimplementing them -- two feature definitions for the same columns is exactly
how the metadata-drift bug etl.py's docstring warns about would happen again.

Cross-sectional and category features (peer_rank_20d, cat_ret_5d, beat_rate_20d,
mkt_ret_1d, breadth_50d, ...) are computed over whatever tickers are present in
the upload: if a ticker's category is known (data/raw/metadata.parquet), its
production peer group is used; unknown tickers are pooled into one catch-all
peer group so those features are still computable rather than left NaN.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import etl  # noqa: E402

# etl.build_features doesn't touch open at all -- ETF signals here are
# high/low/close/volume only, so it isn't required from the upload.
REQUIRED_COLS = ["date", "ticker", "high", "low", "close", "volume"]
COLUMN_ALIASES = {"symbol": "ticker", "adj close": "close", "adj_close": "close"}
NEWS_COLS = ["news_sent_1d", "news_n_1d", "news_sent_5d", "news_n_5d"]
UNKNOWN_CATEGORY = "UPLOADED_PEERS"


class CSVInputError(ValueError):
    """The uploaded CSV doesn't have what's needed to build a feature panel."""


def _normalize_columns(df):
    df = df.rename(columns=lambda c: str(c).strip().lower())
    return df.rename(columns=COLUMN_ALIASES)


def parse_ohlcv_csv(file_obj):
    """Read + validate an uploaded OHLCV CSV.

    Raises CSVInputError (not a raw pandas traceback) for anything a user needs
    to fix in the file itself: missing columns, unparseable dates, duplicate
    (date, ticker) rows.
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        raise CSVInputError(f"could not parse CSV: {e}") from e

    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise CSVInputError(
            f"CSV is missing required column(s): {missing}. expected at least "
            f"{REQUIRED_COLS} (case-insensitive; 'symbol' and 'adj close' are "
            f"also accepted as aliases). optional: 'category'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise CSVInputError("some 'date' values could not be parsed")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    for c in ("high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dupes = int(df.duplicated(subset=["date", "ticker"]).sum())
    if dupes:
        raise CSVInputError(f"{dupes} duplicate (date, ticker) row(s) -- one row per fund per day")

    if df["ticker"].nunique() < 1 or len(df["date"].unique()) < 2:
        raise CSVInputError("need at least 2 distinct dates to compute any trailing feature")

    return df


def _resolve_category(df, tickers):
    """category per ticker: CSV column wins, then the known universe, then a
    catch-all peer group so cross-sectional features are still computable."""
    cat = pd.Series(index=tickers, dtype=object)

    meta_path = ROOT / "data" / "raw" / "metadata.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        if "category" in meta.columns:
            cat.update(meta["category"].reindex(tickers))

    if "category" in df.columns:
        supplied = df.groupby("ticker")["category"].first().reindex(tickers)
        cat.update(supplied.dropna())

    return cat.fillna(UNKNOWN_CATEGORY)


def build_feature_panel(df):
    """Long (date, ticker) x feature DataFrame from a validated OHLCV DataFrame.

    Mirrors dataset_v3's columns exactly (44 etl.FEATURE_COLS + 11 SR + 4 news,
    news filled 0.0 the same way inference.load_feature_panel does) so the result
    can be scored by any registered model without special-casing.
    """
    wide = {col: df.pivot(index="date", columns="ticker", values=col)
            for col in ("high", "low", "close", "volume")}
    high, low, close, volume = wide["high"], wide["low"], wide["close"], wide["volume"]

    cat = _resolve_category(df, close.columns)

    frames, regime = etl.build_features(close, high, low, volume, cat)
    long = etl.features_to_long(frames, regime)

    sr = etl.sr_to_long(etl.build_sr_features(close, high, low))
    long = long.join(sr, how="left")

    for c in NEWS_COLS:
        long[c] = 0.0

    return long.replace([np.inf, -np.inf], np.nan).sort_index()
