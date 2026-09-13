"""Shared fixtures.

Two kinds of test live here. Most run on a small synthetic panel built in-memory,
so `pytest` is fast and works on a fresh clone with no data. A few validate the
real artifacts (`dataset_v3.parquet`, saved models) and are skipped automatically
when those files aren't present -- see the `requires_data` / `requires_models`
markers.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_data: needs data/processed artifacts")
    config.addinivalue_line("markers", "requires_models: needs trained models/")


def pytest_collection_modifyitems(config, items):
    no_data = not (PROC / "dataset_v3.parquet").exists()
    no_models = not any(MODELS.glob("*.cbm")) if MODELS.exists() else True
    for item in items:
        if no_data and "requires_data" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="dataset_v3.parquet not built"))
        if no_models and "requires_models" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="no trained models in models/"))


N_DAYS, N_TICKERS = 400, 12


@pytest.fixture(scope="session")
def synthetic_panel():
    """A deterministic OHLCV panel: geometric random walks with realistic
    intraday ranges, 12 tickers across 3 categories of 4."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS, name="date")
    tickers = [f"ET{i:02d}" for i in range(N_TICKERS)]

    rets = rng.normal(0.0004, 0.011, size=(N_DAYS, N_TICKERS))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    close.columns.name = "ticker"

    span = close * rng.uniform(0.004, 0.02, size=close.shape)
    high = close + span * rng.uniform(0.3, 1.0, size=close.shape)
    low = close - span * rng.uniform(0.3, 1.0, size=close.shape)
    volume = pd.DataFrame(rng.lognormal(12, 0.6, size=close.shape), index=dates, columns=tickers)

    # 2 categories of 6. Group size matters: a rank-based tercile over 4 members
    # splits 1/1/2, so a smaller peer group would make the label look unbalanced
    # for arithmetic reasons rather than because anything is wrong.
    cat = pd.Series([f"cat_{i % 2}" for i in range(N_TICKERS)], index=tickers, name="category")
    return dict(close=close, high=high, low=low, volume=volume, cat=cat)


@pytest.fixture(scope="session")
def synthetic_market_data(synthetic_panel):
    """The same panel in the wide MultiIndex (field x ticker) layout that
    data/raw/market_data.parquet uses."""
    p = synthetic_panel
    return pd.concat({"Close": p["close"], "High": p["high"],
                      "Low": p["low"], "Volume": p["volume"]}, axis=1)


@pytest.fixture(scope="session")
def synthetic_metadata(synthetic_panel):
    cat = synthetic_panel["cat"]
    return pd.DataFrame({"category": cat,
                         "name": [f"Fund {t}" for t in cat.index],
                         "is_leveraged": False,
                         "bad_ticks": False}, index=cat.index)
