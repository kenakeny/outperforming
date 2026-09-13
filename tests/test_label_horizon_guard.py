"""Guard: the dataset's actual label horizon must match the purge horizon.

This is the test that would have caught the bug found on 2026-07-25.

`data/processed/dataset_v3.parquet` had a label of
`close.shift(-25) / close.shift(-5) - 1` -- a 20-trading-day forward return
starting 5 days ahead, so its label window spans t+5 .. t+25. But
`train_models.py` declared `HORIZON = 5` and purged only 5 trading days.
That left **20 trading days of label leakage in every walk-forward fold**, and
every metric produced from that file was inflated.

Nothing caught it because the two halves lived in different places:
  - the label horizon was baked into a **parquet file** nobody re-derived
  - the purge horizon was a **constant in the training script**
and `common.assert_no_overlap` can only verify the horizon it is *told*, so
passing it 5 made it assert a 5-day claim against a 25-day reality. A guard has
to re-measure the horizon from the data itself rather than trust the constant.
"""
import numpy as np
import pandas as pd
import pytest

import etl


def measure_horizon(close, fwd_ret, candidates=range(1, 41), skips=(0, 1, 5, 10),
                    min_overlap=10_000):
    """Recover (horizon, skip) from a stored fwd_ret column by brute force.

    Returns (horizon, skip, correlation) for the best match. An exact
    reconstruction correlates at 1.0; anything materially below that means the
    column wasn't produced from this price panel at all.
    """
    best = (None, None, -np.inf)
    for skip in skips:
        for h in candidates:
            if h <= skip:
                continue
            fwd = (close.shift(-h) / close.shift(-skip) - 1.0).stack()
            fwd.index.names = ["date", "ticker"]
            shared = fwd_ret.index.intersection(fwd.index)
            if len(shared) < min_overlap:
                continue
            corr = fwd_ret.loc[shared].corr(fwd.loc[shared].astype("float64"))
            if corr > best[2]:
                best = (h, skip, corr)
    return best


@pytest.fixture(scope="module")
def panel():
    close, *_ = etl._load_panel(etl.load_config())
    return close


@pytest.mark.slow
@pytest.mark.requires_data
def test_dataset_label_horizon_matches_config(panel):
    """The stored label must be exactly the horizon config.yaml declares, with no
    skip. Any mismatch silently invalidates the purge in every training script."""
    cfg = etl.load_config()
    declared = cfg["labels"]["horizon"]

    stored = pd.read_parquet(etl.PROC / "dataset_v3.parquet",
                             columns=["fwd_ret"])["fwd_ret"].astype("float64")
    horizon, skip, corr = measure_horizon(panel, stored)

    assert corr > 0.999, (
        f"dataset_v3's fwd_ret doesn't reconstruct from the current price panel "
        f"(best corr {corr:.4f} at horizon={horizon}, skip={skip})")
    assert (horizon, skip) == (declared, 0), (
        f"dataset_v3's label is horizon={horizon}, skip={skip}, but config.yaml "
        f"declares horizon={declared}. Every script that purges `declared` trading "
        f"days is leaking {horizon - declared} days of label into its test folds.")


@pytest.mark.slow
@pytest.mark.requires_data
def test_purge_covers_the_measured_label_window(panel):
    """Belt and braces: the purge actually applied must be >= the label window
    measured from the data, expressed in trading days."""
    import common

    cfg = etl.load_config()
    stored = pd.read_parquet(etl.PROC / "dataset_v3.parquet",
                             columns=["fwd_ret"])["fwd_ret"].astype("float64")
    horizon, skip, corr = measure_horizon(panel, stored)
    assert corr > 0.999

    label_window_end = horizon          # label at t reads prices through t+horizon
    purge_used = cfg["labels"]["horizon"]

    idx = panel.index
    first_test = idx[len(idx) // 2]
    cut = common.purge_cutoff(idx, first_test, purge_used)
    gap = idx.searchsorted(first_test) - idx.searchsorted(cut)

    assert gap >= label_window_end, (
        f"purge is {gap} trading days but the label window measured from the data "
        f"spans {label_window_end} -- {label_window_end - gap} days leak into test")


def test_measure_horizon_recovers_a_known_construction(synthetic_panel):
    """The detector itself has to work, or the guard above is theatre."""
    close = synthetic_panel["close"]
    for h, skip in [(5, 0), (10, 0), (25, 5)]:
        fwd = (close.shift(-h) / close.shift(-skip) - 1.0).stack()
        fwd.index.names = ["date", "ticker"]
        got_h, got_skip, corr = measure_horizon(
            close, fwd.dropna().astype("float64"), min_overlap=1_000)
        assert (got_h, got_skip) == (h, skip), f"failed to recover h={h}, skip={skip}"
        assert corr == pytest.approx(1.0, abs=1e-9)
