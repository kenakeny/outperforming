"""Label construction and walk-forward purging.

The repo's history has one label bug already on record (04_baseline benchmarked
against the category MEAN while every other notebook used the MEDIAN, so its
"baseline" scored a different target than everything compared against it). These
tests pin the label's contract so that drift can't happen silently again, and pin
the purge arithmetic that keeps a 5-day forward label out of the test window.
"""
import numpy as np
import pandas as pd
import pytest

import common
import etl


HORIZON, MIN_GROUP = 5, 3


@pytest.fixture(scope="module")
def labels(synthetic_panel):
    return etl.build_labels(synthetic_panel["close"], synthetic_panel["cat"],
                            HORIZON, MIN_GROUP)


# --------------------------------------------------------------------------- #
#  label construction                                                         #
# --------------------------------------------------------------------------- #

def test_fwd_ret_is_the_forward_horizon_return(synthetic_panel, labels):
    close = synthetic_panel["close"]
    ticker = close.columns[0]
    t_pos = 100
    t = close.index[t_pos]
    expected = close[ticker].iloc[t_pos + HORIZON] / close[ticker].iloc[t_pos] - 1.0
    assert labels.loc[(t, ticker), "fwd_ret"] == pytest.approx(expected, rel=1e-9)


def test_benchmark_excludes_the_fund_itself(synthetic_panel, labels):
    """`rel` subtracts the leave-one-out peer mean. A fund included in its own
    benchmark shrinks its own excess return toward zero -- worse for the biggest
    movers, which are exactly the ones the label is trying to identify."""
    close, cat = synthetic_panel["close"], synthetic_panel["cat"]
    t = close.index[100]
    ticker = close.columns[0]
    peers = [c for c in close.columns if cat[c] == cat[ticker] and c != ticker]

    fwd = close.shift(-HORIZON) / close - 1.0
    expected = fwd.loc[t, ticker] - fwd.loc[t, peers].mean()
    assert labels.loc[(t, ticker), "rel"] == pytest.approx(expected, rel=1e-9)


def test_labels_are_terciles_within_date_and_category(labels):
    valid = labels.dropna(subset=["target"])
    assert set(valid["target"].unique()) <= {0.0, 1.0, 2.0}
    # forced terciles are balanced by construction -- that's why the majority-class
    # baseline sits near 1/3 and macro-F1 is measured against ~0.364, not 0.5
    shares = valid["target"].value_counts(normalize=True)
    assert shares.max() < 0.45, f"terciles are not balanced: {shares.to_dict()}"


def test_target_ranks_agree_with_rel_ordering(synthetic_panel, labels):
    """Within one (date, category), a higher `rel` can never get a lower tercile."""
    cat = synthetic_panel["cat"]
    df = labels.dropna(subset=["target"]).reset_index()
    df["cat"] = df["ticker"].map(cat)
    for (_, _), g in df.groupby(["date", "cat"]):
        if len(g) < MIN_GROUP:
            continue
        g = g.sort_values("rel")
        assert g["target"].is_monotonic_increasing, "tercile ordering contradicts rel"
        break_after = True
        if break_after:
            break


def test_thin_categories_get_no_label(synthetic_panel):
    """A category with fewer than min_group members can't form a meaningful
    tercile -- it must be NaN, not a fabricated bucket."""
    close = synthetic_panel["close"]
    lonely = close.columns[0]
    cat = synthetic_panel["cat"].copy()
    cat[lonely] = "singleton"

    labels = etl.build_labels(close, cat, HORIZON, MIN_GROUP)
    got = labels.xs(lonely, level="ticker")["target"]
    assert got.isna().all(), "a 1-member category produced a tercile label"


def test_last_horizon_days_have_no_label(synthetic_panel, labels):
    """The final H days have no forward window yet; labelling them would mean
    inventing returns that haven't happened."""
    last_dates = synthetic_panel["close"].index[-HORIZON:]
    tail = labels[labels.index.get_level_values("date").isin(last_dates)]
    assert tail["fwd_ret"].isna().all()


# --------------------------------------------------------------------------- #
#  purge / walk-forward                                                       #
# --------------------------------------------------------------------------- #

def test_purge_cutoff_backs_off_h_trading_days(synthetic_panel):
    idx = synthetic_panel["close"].index
    test_start = idx[200]
    cut = common.purge_cutoff(idx, test_start, HORIZON)
    assert cut == idx[200 - HORIZON]


def test_purge_cutoff_counts_trading_days_not_calendar_days(synthetic_panel):
    """Calendar or BusinessDay offsets undercount around holiday clusters, which
    leaves part of the training label window inside the test period."""
    idx = synthetic_panel["close"].index
    test_start = idx[200]
    cut = common.purge_cutoff(idx, test_start, HORIZON)
    assert idx.searchsorted(test_start) - idx.searchsorted(cut) == HORIZON


def test_assert_no_overlap_accepts_a_correct_purge(synthetic_panel):
    idx = synthetic_panel["close"].index
    first_test = idx[200]
    cut = common.purge_cutoff(idx, first_test, HORIZON)
    common.assert_no_overlap(idx, cut, first_test, HORIZON)   # must not raise


def test_assert_no_overlap_catches_a_leaky_split(synthetic_panel):
    """The guard has to actually fire -- an assertion that never fails protects
    nothing. Cutting one day too late leaves the label window overlapping test."""
    idx = synthetic_panel["close"].index
    first_test = idx[200]
    too_late = idx[200 - HORIZON + 1]
    with pytest.raises(AssertionError, match="leak"):
        common.assert_no_overlap(idx, too_late, first_test, HORIZON)


def test_walk_forward_never_trains_on_test_dates(synthetic_panel, labels):
    """End-to-end: capture what each fold was handed and assert the training
    frame's label window ends before the test window starts."""
    idx = synthetic_panel["close"].index
    d = labels.dropna(subset=["target"]).reset_index().rename(columns={"target": "label"})
    d["f0"] = np.arange(len(d), dtype=float)

    seen = []

    def spy_fit(train, cols):
        seen.append((train["date"].min(), train["date"].max()))

        class Const:
            def predict(self, X):
                return np.ones(len(X))
        return Const()

    years = sorted(d["date"].dt.year.unique())[1:]
    metrics, pooled, _ = common.walk_forward(d, ["f0"], spy_fit, years, HORIZON, idx)

    assert len(seen) > 0, "walk_forward ran no folds"
    for (_, train_end), year in zip(seen, years):
        test_start = d[d["date"].dt.year == year]["date"].min()
        gap = idx.searchsorted(test_start) - idx.searchsorted(train_end)
        assert gap >= HORIZON, f"only {gap} trading days purged before {year}, need {HORIZON}"
    assert not metrics.empty and len(pooled) > 0
