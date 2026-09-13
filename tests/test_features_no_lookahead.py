"""The tests that matter most: proving no feature peeks at the future.

Every one of the 44 v3 features and 11 S/R features claims to be computable from
data through t. The truncation test is the proof: rebuild the whole feature set
from a panel that has been cut off at date t, and every value at t must equal the
value computed from the full panel. A feature that reads even one future bar
changes when the future is removed.

This is the class of bug that silently inflates every backtest in the repo, and
until now nothing checked for it automatically.
"""
import numpy as np
import pandas as pd
import pytest

import etl


TRUNCATION_DATES = [-40, -80, -150]   # positions from the end of the panel


def _build(panel, upto=None):
    close, high, low, volume = (panel["close"], panel["high"], panel["low"], panel["volume"])
    if upto is not None:
        close, high, low, volume = (df.loc[:upto] for df in (close, high, low, volume))
    frames, regime = etl.build_features(close, high, low, volume, panel["cat"])
    for col in etl.REGIME_COLS:
        frames[col] = pd.DataFrame({t: regime[col] for t in close.columns})
    return frames


@pytest.fixture(scope="module")
def full_features(synthetic_panel):
    return _build(synthetic_panel)


@pytest.mark.parametrize("offset", TRUNCATION_DATES)
def test_features_are_trailing(synthetic_panel, full_features, offset):
    """Truncating the panel at t must not change any feature value at t."""
    t = synthetic_panel["close"].index[offset]
    truncated = _build(synthetic_panel, upto=t)

    drifted = []
    for name in etl.FEATURE_COLS:
        full_row = full_features[name].loc[t]
        trunc_row = truncated[name].loc[t]
        both = full_row.notna() & trunc_row.notna()
        same_values = np.allclose(full_row[both], trunc_row[both], rtol=1e-6, atol=1e-9)
        same_nans = bool((full_row.isna() == trunc_row.isna()).all())
        if not (same_values and same_nans):
            drifted.append(name)

    assert not drifted, f"features changed when the future was removed (lookahead): {drifted}"


@pytest.mark.parametrize("offset", TRUNCATION_DATES)
def test_sr_features_are_trailing(synthetic_panel, offset):
    p = synthetic_panel
    t = p["close"].index[offset]
    full = etl.build_sr_features(p["close"], p["high"], p["low"])
    trunc = etl.build_sr_features(*(df.loc[:t] for df in (p["close"], p["high"], p["low"])))

    drifted = [name for name in full
               if not np.allclose(full[name].loc[t].fillna(-999),
                                  trunc[name].loc[t].fillna(-999), rtol=1e-6, atol=1e-9)]
    assert not drifted, f"S/R features changed under truncation (lookahead): {drifted}"


def test_no_feature_correlates_with_its_own_future_return(synthetic_panel):
    """On random-walk prices there is no signal, so any feature correlating with
    the forward return above noise is reading the future rather than predicting it."""
    p = synthetic_panel
    frames, regime = etl.build_features(p["close"], p["high"], p["low"], p["volume"], p["cat"])
    fwd = (p["close"].shift(-5) / p["close"] - 1.0).stack()

    suspicious = {}
    for name, frame in frames.items():
        joined = pd.concat([frame.stack().rename("f"), fwd.rename("y")], axis=1).dropna()
        if len(joined) < 500 or joined["f"].std() == 0:
            continue
        corr = abs(joined["f"].corr(joined["y"]))
        if corr > 0.10:
            suspicious[name] = round(corr, 4)
    assert not suspicious, f"features correlate with the future on random data: {suspicious}"


def test_sr_resistance_is_a_prior_high(synthetic_panel):
    """dist_res_20 must be measured against the max High over the *previous* 20
    bars. Recomputing one cell by hand catches an off-by-one in the shift."""
    p = synthetic_panel
    sr = etl.build_sr_features(p["close"], p["high"], p["low"], windows=[20])
    t_pos, ticker = 120, p["close"].columns[0]
    t = p["close"].index[t_pos]

    expected_res = p["high"][ticker].iloc[t_pos - 20:t_pos].max()   # excludes t itself
    close_t = p["close"][ticker].loc[t]
    assert sr["dist_res_20"][ticker].loc[t] == pytest.approx(
        (expected_res - close_t) / close_t, rel=1e-9)


def test_sr_range_pos_is_bounded_when_inside_the_channel(synthetic_panel):
    p = synthetic_panel
    sr = etl.build_sr_features(p["close"], p["high"], p["low"])
    for w in (20, 60):
        rp = sr[f"range_pos_{w}"].stack().dropna()
        inside = rp[(sr[f"broke_res_{20}"].stack().dropna().reindex(rp.index) == 0)]
        assert len(inside) > 0
        # position within a support/resistance channel is a ratio, not unbounded
        assert rp.between(-2, 3).mean() > 0.99, f"range_pos_{w} is wildly out of range"


def test_feature_columns_match_the_declared_contract(synthetic_panel):
    """FEATURE_COLS is what the models index by name -- it must describe exactly
    what the builder produces, no more and no less."""
    p = synthetic_panel
    frames, regime = etl.build_features(p["close"], p["high"], p["low"], p["volume"], p["cat"])
    produced = set(frames) | set(regime.columns)
    assert produced == set(etl.FEATURE_COLS), (
        f"missing: {sorted(set(etl.FEATURE_COLS) - produced)}, "
        f"unexpected: {sorted(produced - set(etl.FEATURE_COLS))}")
    assert len(etl.FEATURE_COLS) == 44
