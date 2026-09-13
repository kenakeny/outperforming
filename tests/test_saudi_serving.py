"""serve/saudi.py + the /saudi/* routes.

This record is a *walk-forward evaluation*, not a live model -- every row's
`fwd_ret`/`target` already happened. The tests that matter most here aren't
the read-path mechanics (those mirror inference.py/api.py, already covered
elsewhere); they're the ones that would let a client mistake this for a live
signal, and the ones around the disk cache actually saving the network call
it exists to avoid.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from serve import api, saudi

ARMS = ["saudi_only", "finetune"]
TICKERS = ["9400.SR", "9401.SR", "9402.SR"]
# core.backtest skips any rebalance day with fewer than 6 names, so the
# benchmark tests (which exercise core.backtest through saudi.benchmark) need
# a wider universe than the ranking/history tests do.
BENCH_TICKERS = TICKERS + ["9403.SR", "9404.SR", "9405.SR"]


def _stub_predictions(tickers=TICKERS):
    dates = pd.bdate_range("2025-01-06", periods=4, name="date")
    rng = np.random.default_rng(11)
    rows = []
    for arm in ARMS:
        for d in dates:
            for i, t in enumerate(tickers):
                rows.append({
                    "arm": arm, "date": d, "ticker": t,
                    "score": float(rng.normal(0, 0.4)),
                    "fwd_ret": float(rng.normal(0, 0.02)),
                    "target": int(rng.integers(0, 3)),
                    "pred": (i + (arm == "finetune")) % 3,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def stub_names():
    return {"9400.SR": "FALCOM Saudi Equity ETF", "9401.SR": "FALCOM Petrochemical ETF"}


@pytest.fixture
def patched(monkeypatch, stub_names):
    preds = _stub_predictions()
    for target in (saudi, api.saudi):
        monkeypatch.setattr(target, "load_predictions", lambda: preds)
        monkeypatch.setattr(target, "load_names", lambda: stub_names)
        monkeypatch.setattr(target, "available", lambda: True)
    return preds


@pytest.fixture
def client(patched):
    return TestClient(api.app)


# --------------------------------------------------------------------------- #
#  this is a record, not a live signal -- the part worth over-testing        #
# --------------------------------------------------------------------------- #

def test_every_saudi_response_flags_itself_as_realized_history(client):
    """The one invariant that can't regress silently: a client that stops
    checking this flag would present a graded-already outcome as a live call."""
    for path, params in [
        ("/saudi/meta", {}),
        ("/saudi/signals", {"arm": "saudi_only"}),
        ("/saudi/tickers/9400.SR", {"arm": "saudi_only"}),
    ]:
        body = client.get(path, params=params).json()
        assert body["is_realized_history"] is True, f"{path} dropped the history flag"


def test_signals_carries_the_realized_outcome_next_to_the_score(client):
    """Hiding fwd_ret/target would make a graded row look like an open call."""
    body = client.get("/saudi/signals", params={"arm": "saudi_only"}).json()
    row = body["signals"][0]
    assert {"score", "prediction", "realized_class", "fwd_ret"} <= set(row)


# --------------------------------------------------------------------------- #
#  ranking, dates, arms                                                       #
# --------------------------------------------------------------------------- #

def test_signals_default_to_the_latest_date_and_rank_by_score(client, patched):
    body = client.get("/saudi/signals", params={"arm": "saudi_only"}).json()
    assert body["date"] == str(patched["date"].max().date())
    scores = [s["score"] for s in body["signals"]]
    assert scores == sorted(scores, reverse=True)
    assert body["n"] == len(TICKERS)


def test_signals_can_be_asked_for_an_earlier_evaluation_date(client, patched):
    earlier = str(sorted(patched["date"].unique())[0].date())
    body = client.get("/saudi/signals", params={"arm": "saudi_only", "date": earlier}).json()
    assert body["date"] == earlier


def test_signals_404s_on_a_date_outside_the_evaluation_record(client):
    r = client.get("/saudi/signals", params={"arm": "saudi_only", "date": "1999-01-04"})
    assert r.status_code == 404


def test_signals_404s_on_an_unknown_arm(client):
    assert client.get("/saudi/signals", params={"arm": "not_an_arm"}).status_code == 404


def test_meta_lists_only_arms_actually_present(client, patched):
    body = client.get("/saudi/meta").json()
    assert set(body["arms"]) == set(ARMS)
    assert body["horizon_days"] == saudi.HORIZON
    assert {"ticker", "name"} <= set(body["funds"][0])


def test_meta_resolves_names_and_falls_back_to_the_ticker(client):
    """9402.SR has no entry in stub_names -- it must not disappear or 500."""
    funds = {f["ticker"]: f["name"] for f in client.get("/saudi/meta").json()["funds"]}
    assert funds["9400.SR"] == "FALCOM Saudi Equity ETF"
    assert funds["9402.SR"] == "9402.SR"


def test_ticker_history_is_scoped_to_one_ticker_and_arm(client):
    body = client.get("/saudi/tickers/9400.sr", params={"arm": "finetune", "days": 2}).json()
    assert body["ticker"] == "9400.SR"
    assert body["name"] == "FALCOM Saudi Equity ETF"
    assert len(body["history"]) == 2


def test_ticker_history_404s_on_a_ticker_outside_the_record(client):
    assert client.get("/saudi/tickers/ZZZZ").status_code == 404


def test_saudi_routes_404_cleanly_when_no_predictions_file_exists(monkeypatch):
    monkeypatch.setattr(api.saudi, "available", lambda: False)
    body = TestClient(api.app).get("/saudi/signals", params={"arm": "saudi_only"})
    assert body.status_code == 404
    assert "saudi.py" in body.json()["detail"]


# --------------------------------------------------------------------------- #
#  the benchmark endpoint: caching is the point                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def bench_ready(monkeypatch, tmp_path):
    """core.backtest drops any rebalance day under 6 names, so this needs its
    own wider stub -- unlike `patched`, which only needs enough for ranking."""
    preds = _stub_predictions(BENCH_TICKERS)
    monkeypatch.setattr(saudi, "load_predictions", lambda: preds)
    pred_path = tmp_path / "preds.parquet"
    pred_path.write_bytes(b"v1")
    monkeypatch.setattr(saudi, "PRED_PATH", pred_path)
    monkeypatch.setattr(saudi, "BENCH_CACHE", tmp_path / "bench_cache.json")
    return pred_path


def test_benchmark_fetches_the_network_once_and_shares_it_across_arms(monkeypatch, bench_ready):
    """The whole reason for the disk cache: TASI/FALCOM don't vary by arm, and
    a network round trip per arm would be four times the cost for identical data."""
    calls = {"n": 0}

    def fake_fetch(dates):
        calls["n"] += 1
        return {"TASI": np.zeros(len(dates))}

    monkeypatch.setattr(saudi, "_fetch_benchmarks", fake_fetch)

    saudi.benchmark("saudi_only")
    saudi.benchmark("finetune")
    assert calls["n"] == 1, "second arm re-fetched instead of reusing the cached benchmark"


def test_benchmark_response_has_a_curve_and_stats_per_series(monkeypatch, bench_ready):
    monkeypatch.setattr(saudi, "_fetch_benchmarks",
                        lambda dates: {"TASI": np.linspace(-0.01, 0.01, len(dates))})

    out = saudi.benchmark("saudi_only")
    assert set(out["series"]) == {"model", "TASI"}
    assert set(out["stats"]) == {"model", "TASI"}
    assert {"ann_return", "ir", "hit_rate", "n"} <= set(out["stats"]["model"])
    # 4 dates with a 20-day rebalance horizon collapses to a single rebalance
    # point in this stub -- real data (480 dates) has many; the shape, not the
    # count, is what this test is pinning.
    assert len(out["series"]["model"]) >= 1
    assert {"date", "value"} <= set(out["series"]["model"][0])


def test_benchmark_cache_invalidates_when_the_predictions_file_changes(monkeypatch, bench_ready):
    calls = {"n": 0}

    def fake_fetch(dates):
        calls["n"] += 1
        return {"TASI": np.zeros(len(dates))}

    monkeypatch.setattr(saudi, "_fetch_benchmarks", fake_fetch)
    saudi.benchmark("saudi_only")

    bench_ready.write_bytes(b"v2 -- saudi.py was re-run")
    saudi.benchmark("saudi_only")
    assert calls["n"] == 2, "a changed predictions file should invalidate the cached benchmark"
