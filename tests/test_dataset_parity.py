"""Parity between etl.py and the notebook-produced dataset_v3.parquet.

etl.py claims to be a faithful port of notebooks/03_features.ipynb. This test is
the evidence for that claim, run against the real panel.

It splits the 44 features into two groups, because they behave differently:

  PER_TICKER (32 features) -- computed from one fund's own OHLCV. These must match
  dataset_v3 *exactly*; any drift is a porting bug.

  UNIVERSE_DEPENDENT (12 features) -- cross-sectional ranks, category aggregates and
  market-wide regime series. These depend on *which funds are in the panel* and
  *what category each is in*, both of which come from `metadata.parquet`. That file
  is re-downloaded periodically and financedatabase's categories are not
  point-in-time, so a rebuild legitimately shifts these features even though the
  code is identical. They're checked for high correlation, not equality.

That second group is the point of this test: re-running the ETL after a metadata
refresh silently changes 12 features *and the label*, because the peer group
itself moved. Models trained before and after a refresh aren't strictly
comparable, and nothing else in the repo says so.

Slow (rebuilds the full feature panel). Run with: pytest -m slow
"""
import numpy as np
import pandas as pd
import pytest

import etl


# features that depend on the panel's membership or on category assignment
UNIVERSE_DEPENDENT = {
    "cs_rank_ret_5d", "cs_rank_ret_20d", "cs_rank_rsi", "cs_rank_volatility",
    "peer_rank_20d", "d_peer_rank_5d", "beta_60d", "cat_ret_5d", "cat_disp_5d",
    "beat_rate_20d",
    # market-wide series: means/counts taken across whatever funds are in the panel
    "mkt_ret_1d", "mkt_vol_20d", "mkt_vol_zscore",
}

# willr_14 / stoch_k / stoch_d divide by a 14-day high-low range; on a near-flat
# window that denominator is tiny and the result is numerically unstable, so a
# handful of rows out of millions can differ materially. Compared on rank instead.
UNSTABLE_DENOMINATOR = {"willr_14", "stoch_k", "stoch_d"}

# vol_zscore normalizes against a rolling 252-day window with min_periods=60. For a
# fund with interior NaN gaps (trading halts, missing bars) the number of valid
# observations inside that window depends on the panel's exact date rows, so it can
# resolve differently between two builds. Measured: 9,464 of 3.2M rows differ, across
# 74 tickers -- and 100% of those tickers have interior NaN gaps, versus 4.7% of the
# universe. So the discrepancy is fully explained by gappy funds, and the feature is
# required to reproduce exactly on funds with continuous history.
GAP_SENSITIVE = {"vol_zscore"}

PER_TICKER = [c for c in etl.FEATURE_COLS if c not in UNIVERSE_DEPENDENT]
EXACT = [c for c in PER_TICKER if c not in UNSTABLE_DENOMINATOR | GAP_SENSITIVE]


@pytest.fixture(scope="module")
def rebuilt():
    cfg = etl.load_config()
    close, high, low, volume, cat = etl._load_panel(cfg)
    frames, regime = etl.build_features(close, high, low, volume, cat)
    return etl.features_to_long(frames, regime)


#: The notebook-era artifact, preserved when etl.py first rebuilt dataset_v3 on
#: 2026-07-25. Parity must be measured against *that* file -- comparing against the
#: current dataset_v3.parquet would be circular, since etl.py now produces it.
#: Only its FEATURE columns are trusted: its label was a horizon-25/skip-5 forward
#: return (see tests/test_label_horizon_guard.py), which is why it was replaced.
LEGACY = etl.PROC / "dataset_v3_pre_rebuild.parquet"


@pytest.fixture(scope="module")
def stored():
    if not LEGACY.exists():
        pytest.skip(f"{LEGACY.name} not present -- nothing to check parity against")
    return pd.read_parquet(LEGACY, columns=etl.FEATURE_COLS)


@pytest.fixture(scope="module")
def aligned(rebuilt, stored):
    idx = stored.index.intersection(rebuilt.index)
    assert len(idx) > 0.99 * len(stored), (
        f"rebuilt panel only covers {len(idx)}/{len(stored)} of dataset_v3's rows -- "
        "the ETL is producing a different universe, not just different values")
    return stored.loc[idx].astype("float64"), rebuilt.loc[idx].astype("float64")


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("feature", EXACT)
def test_per_ticker_feature_reproduces_exactly(aligned, feature):
    old, new = aligned
    both = old[feature].notna() & new[feature].notna()
    assert both.mean() > 0.5, f"{feature} is mostly NaN in one of the two builds"
    assert np.allclose(old[feature][both], new[feature][both], rtol=1e-4, atol=1e-6), (
        f"{feature} differs from dataset_v3 -- porting bug in etl.build_features "
        f"(max abs diff {(old[feature][both] - new[feature][both]).abs().max():.6g})")


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("feature", sorted(UNSTABLE_DENOMINATOR))
def test_unstable_denominator_features_agree_in_rank(aligned, feature):
    old, new = aligned
    both = old[feature].notna() & new[feature].notna()
    assert old[feature][both].corr(new[feature][both]) > 0.999


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("feature", sorted(GAP_SENSITIVE))
def test_gap_sensitive_feature_reproduces_on_funds_with_continuous_history(aligned, feature):
    """Every mismatch must be attributable to a fund with interior NaN gaps. If one
    turns up on a fund with continuous history, min_periods bookkeeping is not the
    explanation and the port has a real bug."""
    close, *_ = etl._load_panel(etl.load_config())
    interior_gaps = close.apply(
        lambda s: s.loc[s.first_valid_index():s.last_valid_index()].isna().sum())
    gappy = set(interior_gaps[interior_gaps > 0].index)
    assert len(gappy) < 0.2 * close.shape[1], "gappy funds should be the exception"

    old, new = aligned
    both = old[feature].notna() & new[feature].notna()
    diff = (old[feature][both] - new[feature][both]).abs()
    mismatched = diff[diff > 1e-4]

    assert len(mismatched) < 0.01 * len(diff), (
        f"{feature} differs on {len(mismatched)/len(diff):.2%} of rows -- too many to "
        "be explained by gappy funds alone")
    offenders = set(mismatched.index.get_level_values("ticker")) - gappy
    assert not offenders, (
        f"{feature} differs on funds with continuous history: {sorted(offenders)[:10]} "
        "-- that's a porting bug, not a min_periods artifact")


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("feature", sorted(UNIVERSE_DEPENDENT))
def test_universe_dependent_feature_stays_highly_correlated(aligned, feature):
    """These may shift when metadata is refreshed, but a *large* move means the
    peer groups have changed enough that previously reported metrics no longer
    describe the current dataset."""
    old, new = aligned
    both = old[feature].notna() & new[feature].notna()
    corr = old[feature][both].corr(new[feature][both])
    assert corr > 0.85, (
        f"{feature} correlates only {corr:.3f} with dataset_v3. Either the port is "
        "wrong or the fund universe/category map has drifted far enough that models "
        "trained on dataset_v3 should be retrained before their metrics are quoted.")
