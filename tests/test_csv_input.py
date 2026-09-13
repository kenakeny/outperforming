"""serve/csv_input.py -- turning an uploaded OHLCV CSV into a scorable panel.

Feature values themselves are exercised end-to-end via etl.build_features'
own tests; what's specific to this module is CSV validation and how peer
categories get resolved for tickers the pipeline has never seen.
"""
import io

import pandas as pd
import pytest

import etl
from serve import csv_input


def _ohlcv_csv_text(close, high, low, volume, category=None):
    """Long-format OHLCV CSV text from wide (date x ticker) frames."""
    long = pd.concat({"high": high.stack(), "low": low.stack(),
                      "close": close.stack(), "volume": volume.stack()}, axis=1)
    long.index.names = ["date", "ticker"]
    df = long.reset_index()
    if category is not None:
        df["category"] = df["ticker"].map(category)
    return df.to_csv(index=False)


# --------------------------------------------------------------------------- #
#  parsing / validation                                                       #
# --------------------------------------------------------------------------- #

def test_parse_ohlcv_csv_accepts_column_aliases_and_normalizes():
    text = ("Date,Symbol,High,Low,Adj Close,Volume\n"
            "2024-01-01,aaa,10.5,9.5,10.0,1000\n"
            "2024-01-02,aaa,11.0,10.0,10.5,1100\n")
    df = csv_input.parse_ohlcv_csv(io.StringIO(text))
    assert list(df["ticker"]) == ["AAA", "AAA"]
    assert df["close"].tolist() == [10.0, 10.5]


def test_parse_ohlcv_csv_rejects_missing_columns():
    text = "date,ticker,close\n2024-01-01,AAA,10\n2024-01-02,AAA,11\n"
    with pytest.raises(csv_input.CSVInputError, match="missing required column"):
        csv_input.parse_ohlcv_csv(io.StringIO(text))


def test_parse_ohlcv_csv_rejects_unparseable_dates():
    text = "date,ticker,high,low,close,volume\nnot-a-date,AAA,11,9,10,1000\n"
    with pytest.raises(csv_input.CSVInputError, match="date"):
        csv_input.parse_ohlcv_csv(io.StringIO(text))


def test_parse_ohlcv_csv_rejects_duplicate_date_ticker_rows():
    text = ("date,ticker,high,low,close,volume\n"
            "2024-01-01,AAA,11,9,10,1000\n"
            "2024-01-01,AAA,11,9,10,1000\n"
            "2024-01-02,AAA,11,9,10,1000\n")
    with pytest.raises(csv_input.CSVInputError, match="duplicate"):
        csv_input.parse_ohlcv_csv(io.StringIO(text))


def test_parse_ohlcv_csv_rejects_a_single_date():
    text = "date,ticker,high,low,close,volume\n2024-01-01,AAA,11,9,10,1000\n"
    with pytest.raises(csv_input.CSVInputError, match="2 distinct dates"):
        csv_input.parse_ohlcv_csv(io.StringIO(text))


# --------------------------------------------------------------------------- #
#  feature panel construction                                                 #
# --------------------------------------------------------------------------- #

def test_build_feature_panel_has_every_production_column(synthetic_panel, monkeypatch, tmp_path):
    """Same columns dataset_v3 + sr_features + news_features have, so any
    registered model can score an upload without a special case."""
    monkeypatch.setattr(csv_input, "ROOT", tmp_path)  # no metadata.parquet here
    p = synthetic_panel
    text = _ohlcv_csv_text(p["close"], p["high"], p["low"], p["volume"])
    df = csv_input.parse_ohlcv_csv(io.StringIO(text))
    panel = csv_input.build_feature_panel(df)

    expected = set(etl.FEATURE_COLS) | {
        "dist_res_20", "dist_sup_20", "range_pos_20", "sr_width_20", "res_touch_20",
        "sup_touch_20", "broke_res_20", "broke_sup_20",
        "dist_res_60", "dist_sup_60", "range_pos_60"} | set(csv_input.NEWS_COLS)
    assert set(panel.columns) == expected
    assert (panel[csv_input.NEWS_COLS] == 0.0).all().all()
    assert panel.index.names == ["date", "ticker"]


def test_build_feature_panel_pools_unknown_tickers_into_one_peer_group(synthetic_panel, monkeypatch, tmp_path):
    """Cross-sectional / category features must still compute for tickers with
    no known category, not go all-NaN -- see UNKNOWN_CATEGORY."""
    monkeypatch.setattr(csv_input, "ROOT", tmp_path)
    p = synthetic_panel
    text = _ohlcv_csv_text(p["close"], p["high"], p["low"], p["volume"])
    df = csv_input.parse_ohlcv_csv(io.StringIO(text))
    panel = csv_input.build_feature_panel(df)
    last_date = panel.index.get_level_values("date").max()
    row = panel.xs(last_date, level="date")
    assert row["beat_rate_20d"].notna().any()
    assert row["peer_rank_20d"].notna().any()


def test_resolve_category_prefers_known_metadata_over_pooling(synthetic_panel, monkeypatch, tmp_path):
    """A ticker recognized from data/raw/metadata.parquet uses its real peer
    group instead of being pooled with every other uploaded ticker."""
    p = synthetic_panel
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    known = list(p["cat"].index[:2])
    meta = pd.DataFrame({"category": ["known_cat_a", "known_cat_b"]},
                        index=pd.Index(known, name="ticker"))
    meta.to_parquet(raw / "metadata.parquet")
    monkeypatch.setattr(csv_input, "ROOT", tmp_path)

    text = _ohlcv_csv_text(p["close"], p["high"], p["low"], p["volume"])
    df = csv_input.parse_ohlcv_csv(io.StringIO(text))
    cat = csv_input._resolve_category(df, p["close"].columns)
    assert cat[known[0]] == "known_cat_a"
    assert cat[known[1]] == "known_cat_b"
    assert (cat.drop(index=known) == csv_input.UNKNOWN_CATEGORY).all()


def test_resolve_category_lets_csv_supplied_category_override_metadata(synthetic_panel, monkeypatch, tmp_path):
    monkeypatch.setattr(csv_input, "ROOT", tmp_path)
    p = synthetic_panel
    text = _ohlcv_csv_text(p["close"], p["high"], p["low"], p["volume"], category=p["cat"])
    df = csv_input.parse_ohlcv_csv(io.StringIO(text))
    cat = csv_input._resolve_category(df, p["close"].columns)
    assert (cat == p["cat"]).all()
