"""Inference contract and API surface.

The serving tests use a stub predictor and a stub feature panel so they run
without the 3.3M-row parquet or a GPU; the tests that exercise the real artifacts
are marked `requires_models` and skip when `models/` is empty.
"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import inference
from serve import api


FEATS = ["ret_1d", "ret_5d", "vol_20d"]
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


class StubModel:
    """Deterministic 3-class probabilities so assertions can be exact."""
    def __init__(self, feature_names=FEATS):
        self.feature_names_ = feature_names

    def predict_proba(self, X):
        n = len(X)
        base = np.linspace(0.1, 0.6, n)
        return np.column_stack([0.7 - base, np.full(n, 0.3), base])


@pytest.fixture
def stub_panel():
    dates = pd.bdate_range("2026-01-01", periods=6, name="date")
    idx = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])
    rng = np.random.default_rng(3)
    return pd.DataFrame(rng.normal(size=(len(idx), len(FEATS))), index=idx, columns=FEATS)


@pytest.fixture
def stub_meta():
    """Metadata with every column the lookup layer filters on, so the facet
    tests exercise the same join the real deployment does."""
    return pd.DataFrame({"name": [f"Fund {t}" for t in TICKERS],
                         "category": ["cat_a", "cat_a", "cat_b", "cat_b"],
                         "category_group": ["grp_1", "grp_1", "grp_1", "grp_2"],
                         "family": ["Acme", "Acme", "Beta Co", "Beta Co"],
                         "exchange": ["PCX", "PCX", "NMS", "NMS"],
                         "is_leveraged": [False, True, False, False]},
                        index=pd.Index(TICKERS, name="ticker"))


@pytest.fixture
def patched(monkeypatch, stub_panel, stub_meta):
    monkeypatch.setattr(inference, "load_feature_panel", lambda: stub_panel)
    monkeypatch.setattr(inference, "load_metadata", lambda: stub_meta)
    monkeypatch.setattr(inference, "latest_date", lambda: stub_panel.index.get_level_values("date").max())
    predictor = inference.Predictor("stub", StubModel(), FEATS, "stub")
    monkeypatch.setattr(api, "get_predictor", lambda name: predictor)
    monkeypatch.setattr(inference, "available_models", lambda: {"stub": "stub.cbm"})
    monkeypatch.setattr(inference, "load_results", lambda: {"CatBoost +sr": {"macro_f1": 0.4351}})
    return predictor


# --------------------------------------------------------------------------- #
#  inference contract                                                         #
# --------------------------------------------------------------------------- #

def test_predict_frame_rejects_a_missing_feature(patched, stub_panel):
    """Scoring against the wrong feature set yields plausible-looking numbers that
    mean nothing -- it has to fail loudly, not zero-fill."""
    with pytest.raises(ValueError, match="missing"):
        patched.predict_frame(stub_panel.drop(columns=["vol_20d"]))


def test_predict_frame_reorders_columns_to_training_order(patched, stub_panel):
    """Tree models index features positionally once loaded; a shuffled frame must
    still be scored in the order the model was trained on."""
    shuffled = stub_panel[FEATS[::-1]]
    assert patched.predict_frame(shuffled).equals(patched.predict_frame(stub_panel))


def test_predict_date_scores_every_fund_and_ranks_by_score(patched, stub_panel):
    date = stub_panel.index.get_level_values("date")[0]
    out = patched.predict_date(str(date.date()))
    assert len(out) == len(TICKERS)
    assert out["score"].is_monotonic_decreasing, "results are not ranked by conviction"
    assert set(out["prediction"]) <= set(inference.CLASS_NAMES.values())


def test_score_is_outperform_minus_underperform(patched, stub_panel):
    date = stub_panel.index.get_level_values("date")[0]
    out = patched.predict_date(str(date.date()))
    assert np.allclose(out["score"], out["outperform"] - out["underperform"])
    assert out["score"].between(-1, 1).all()


def test_predict_date_joins_fund_metadata(patched, stub_panel):
    date = stub_panel.index.get_level_values("date")[0]
    out = patched.predict_date(str(date.date()))
    assert {"name", "category"} <= set(out.columns)
    assert out["name"].notna().all()


def test_predict_date_honours_top_n_and_min_probability(patched, stub_panel):
    date = stub_panel.index.get_level_values("date")[0]
    assert len(patched.predict_date(str(date.date()), top_n=2)) == 2
    filtered = patched.predict_date(str(date.date()), min_probability=0.5)
    assert (filtered["confidence"] >= 0.5).all()


def test_predict_date_rejects_a_date_with_no_data(patched):
    with pytest.raises(KeyError, match="no data"):
        patched.predict_date("1999-01-04")


def test_unknown_model_name_is_a_keyerror():
    with pytest.raises(KeyError, match="unknown model"):
        inference.Predictor.load("not_a_model")


# --------------------------------------------------------------------------- #
#  API surface                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def client(patched):
    return TestClient(api.app)


def test_health_reports_loaded_models_and_latest_date(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_available"] == ["stub"]
    assert body["latest_date"] == "2026-01-08"


def test_signals_defaults_to_the_latest_date(client):
    body = client.get("/signals").json()
    assert body["date"] == "2026-01-08"
    assert len(body["signals"]) == len(TICKERS)
    assert "ticker" in body["signals"][0]


def test_signals_respects_top_n(client):
    body = client.get("/signals", params={"top_n": 2}).json()
    assert len(body["signals"]) == 2
    assert body["n_scored"] == len(TICKERS)


def test_signals_filters_by_category(client):
    body = client.get("/signals", params={"category": "cat_b"}).json()
    assert {s["category"] for s in body["signals"]} == {"cat_b"}


def test_signals_404s_on_an_empty_category(client):
    assert client.get("/signals", params={"category": "nope"}).status_code == 404


def test_signals_404s_on_a_date_with_no_data(client):
    r = client.get("/signals", params={"date": "1999-01-04"})
    assert r.status_code == 404


def test_signals_rejects_an_out_of_range_top_n(client):
    assert client.get("/signals", params={"top_n": 0}).status_code == 422
    assert client.get("/signals", params={"top_n": 10_000}).status_code == 422


def test_picks_returns_the_top_n_by_score(client):
    body = client.get("/picks", params={"n": 2}).json()
    assert body["date"] == "2026-01-08"
    assert len(body["buy"]) == 2
    assert {"ticker", "score"} <= set(body["buy"][0])
    scores = [row["score"] for row in body["buy"]]
    assert scores == sorted(scores, reverse=True)


def test_picks_404s_on_a_date_with_no_data(client):
    assert client.get("/picks", params={"date": "1999-01-04"}).status_code == 404


def test_models_explains_the_macro_f1_convention(client):
    body = client.get("/models").json()
    assert "0.364" in body["metric_convention"], (
        "forced-tercile macro-F1 is not comparable to argmax-classification F1; "
        "the response has to say which convention it reports")


def test_ticker_history_returns_a_dated_score_series(client):
    body = client.get("/tickers/aaa", params={"days": 4}).json()
    assert body["ticker"] == "AAA"
    assert len(body["history"]) == 4
    assert {"date", "score", "outperform"} <= set(body["history"][0])


def test_ticker_history_404s_on_an_unknown_ticker(client):
    assert client.get("/tickers/ZZZZ").status_code == 404


# --------------------------------------------------------------------------- #
#  lookup: search, filter, sort, paginate                                     #
# --------------------------------------------------------------------------- #

def test_scored_date_is_cached_between_calls(patched, stub_panel):
    """The screener re-queries on every keystroke and page click. If each one
    re-ran the model over the universe the UI would be unusable, so the scored
    frame per (predictor, date) is cached."""
    date = str(stub_panel.index.get_level_values("date")[0].date())
    calls = []
    original = patched.model.predict_proba
    patched.model.predict_proba = lambda X: (calls.append(1), original(X))[1]

    patched.scored_date(date)
    patched.scored_date(date)
    assert len(calls) == 1, "the second call re-scored instead of using the cache"


def test_scored_date_returns_a_copy_so_callers_cannot_poison_the_cache(patched, stub_panel):
    date = str(stub_panel.index.get_level_values("date")[0].date())
    patched.scored_date(date)["score"] = 999.0
    assert (patched.scored_date(date)["score"] != 999.0).all()


def test_dates_lists_scoreable_days_newest_first(client):
    body = client.get("/dates").json()
    assert body["latest"] == "2026-01-08"
    assert body["dates"][0] == "2026-01-08"
    assert body["dates"] == sorted(body["dates"], reverse=True)


def test_signals_free_text_matches_ticker_or_name(client):
    assert {s["ticker"] for s in client.get("/signals", params={"q": "ccc"}).json()["signals"]} == {"CCC"}
    assert len(client.get("/signals", params={"q": "Fund"}).json()["signals"]) == len(TICKERS)


def test_signals_accepts_several_values_for_one_facet(client):
    body = client.get("/signals", params=[("category", "cat_a"), ("category", "cat_b")]).json()
    assert body["n_scored"] == len(TICKERS)


def test_signals_filters_stack_across_facets(client):
    body = client.get("/signals", params={"category_group": "grp_1", "exchange": "NMS"}).json()
    assert {s["ticker"] for s in body["signals"]} == {"CCC"}


def test_signals_filters_by_predicted_class(client):
    body = client.get("/signals", params={"prediction": "outperform"}).json()
    assert all(s["prediction"] == "outperform" for s in body["signals"])


def test_signals_can_exclude_leveraged_products(client):
    body = client.get("/signals", params={"exclude_leveraged": True}).json()
    assert "BBB" not in {s["ticker"] for s in body["signals"]}


def test_signals_sorts_by_the_requested_column_and_direction(client):
    scores = [s["score"] for s in client.get("/signals", params={"sort": "score", "order": "asc"}).json()["signals"]]
    assert scores == sorted(scores)
    names = [s["ticker"] for s in client.get("/signals", params={"sort": "ticker", "order": "asc"}).json()["signals"]]
    assert names == sorted(names)


def test_signals_rejects_sorting_by_an_arbitrary_column(client):
    """Sorting by a raw feature isn't a contract worth supporting -- and letting
    any column name through makes the response shape depend on the panel."""
    assert client.get("/signals", params={"sort": "ret_1d"}).status_code == 422


def test_signals_pages_without_dropping_or_repeating_rows(client):
    first = client.get("/signals", params={"top_n": 2, "offset": 0}).json()
    second = client.get("/signals", params={"top_n": 2, "offset": 2}).json()
    seen = [s["ticker"] for s in first["signals"]] + [s["ticker"] for s in second["signals"]]
    assert sorted(seen) == sorted(TICKERS)
    assert first["n_scored"] == second["n_scored"] == len(TICKERS)


def test_signals_summary_describes_the_filtered_set_not_the_universe(client):
    body = client.get("/signals", params={"category": "cat_b"}).json()
    assert body["summary"]["n"] == 2
    assert body["universe_scored"] == len(TICKERS)


def test_signals_reports_a_filter_it_could_not_honour(client, monkeypatch, stub_meta):
    """A filter silently dropping every row is indistinguishable from a genuine
    empty result, so an unhonourable filter has to come back named."""
    monkeypatch.setattr(inference, "load_metadata", lambda: stub_meta.drop(columns=["family"]))
    body = client.get("/signals", params={"family": "Acme"}).json()
    assert body["ignored_filters"] == ["family"]


def test_an_empty_but_valid_filter_combination_is_not_an_error(client):
    """No fund is both cat_a and grp_2, but each value exists on its own --
    that's a real answer ('nothing matches'), unlike a typo'd category name,
    which stays a 404."""
    body = client.get("/signals", params={"category": "cat_a", "category_group": "grp_2"})
    assert body.status_code == 200
    assert body.json()["signals"] == []


def test_search_ranks_an_exact_ticker_first(client):
    body = client.get("/search", params={"q": "AAA"}).json()
    assert body["results"][0]["ticker"] == "AAA"


def test_search_marks_hits_that_are_not_in_the_scored_panel(client):
    body = client.get("/search", params={"q": "AAA"}).json()
    assert "scored" in body["results"][0]


def test_search_requires_a_query(client):
    assert client.get("/search").status_code == 422


def test_facets_expose_the_filter_vocabulary(client):
    body = client.get("/facets").json()
    assert body["fields"] == list(api.universe.FACET_FIELDS)
    assert set(body["predictions"]) == set(inference.CLASS_NAMES.values())


def test_categories_rolls_up_by_peer_group(client):
    body = client.get("/categories", params={"min_funds": 1}).json()
    assert body["field"] == "category"
    groups = {g["category"]: g for g in body["groups"]}
    assert set(groups) == {"cat_a", "cat_b"}
    assert groups["cat_a"]["n_funds"] == 2
    means = [g["mean_score"] for g in body["groups"]]
    assert means == sorted(means, reverse=True)


def test_categories_drops_groups_below_the_size_floor(client):
    assert client.get("/categories", params={"min_funds": 3}).json()["groups"] == []


def test_categories_rejects_grouping_by_an_unknown_field(client):
    assert client.get("/categories", params={"field": "score"}).status_code == 422


def test_ticker_detail_carries_metadata_and_the_latest_row(client):
    body = client.get("/tickers/aaa", params={"days": 4}).json()
    assert body["latest"]["date"] == body["history"][-1]["date"]
    assert body["n_points"] == 4


# --------------------------------------------------------------------------- #
#  CSV upload (/predict/csv)                                                  #
# --------------------------------------------------------------------------- #

def _small_ohlcv_csv(tickers=("AAA", "BBB"), n_days=5):
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rows = []
    for t in tickers:
        price = 100.0
        for d in dates:
            price *= 1.01
            rows.append((d.date(), t, price * 1.01, price * 0.99, price, 10_000))
    df = pd.DataFrame(rows, columns=["date", "ticker", "high", "low", "close", "volume"])
    return df.to_csv(index=False)


def test_predict_csv_scores_uploaded_data_and_defaults_to_latest_date_only(client):
    r = client.post("/predict/csv", params={"model": "stub"},
                    files={"file": ("prices.csv", io.BytesIO(_small_ohlcv_csv().encode()), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_scored"] == 2   # 1 row per ticker, latest date only
    assert {"date", "ticker", "score", "prediction", "confidence"} <= set(body["predictions"][0])


def test_predict_csv_latest_only_false_scores_every_row(client):
    r = client.post("/predict/csv", params={"model": "stub", "latest_only": False},
                    files={"file": ("prices.csv", io.BytesIO(_small_ohlcv_csv().encode()), "text/csv")})
    assert r.json()["n_scored"] > 2


def test_predict_csv_honours_top_n(client):
    r = client.post("/predict/csv", params={"model": "stub", "latest_only": False, "top_n": 3},
                    files={"file": ("prices.csv", io.BytesIO(_small_ohlcv_csv().encode()), "text/csv")})
    assert len(r.json()["predictions"]) == 3


def test_predict_csv_400s_on_a_csv_missing_required_columns(client):
    bad = b"not,a,valid,ohlcv,csv\n1,2,3,4,5\n"
    r = client.post("/predict/csv", params={"model": "stub"},
                    files={"file": ("bad.csv", io.BytesIO(bad), "text/csv")})
    assert r.status_code == 400
    assert "missing required column" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  real artifacts                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.requires_models
def test_saved_models_declare_the_features_the_pipeline_builds():
    """A model artifact whose feature names have drifted from etl.FEATURE_COLS +
    S/R can't be served by the pipeline's output at all."""
    import etl
    known = set(etl.FEATURE_COLS) | {
        "dist_res_20", "dist_sup_20", "range_pos_20", "sr_width_20", "res_touch_20",
        "sup_touch_20", "broke_res_20", "broke_sup_20",
        "dist_res_60", "dist_sup_60", "range_pos_60",
        "news_sent_1d", "news_n_1d", "news_sent_5d", "news_n_5d"}
    for name in inference.available_models():
        predictor = inference.Predictor.load(name)
        unknown = set(predictor.feature_names) - known
        assert not unknown, f"{name} was trained on unknown features: {sorted(unknown)}"


@pytest.mark.requires_models
@pytest.mark.requires_data
def test_real_predictor_scores_the_latest_date():
    names = sorted(inference.available_models())
    predictor = inference.Predictor.load(names[0])
    out = predictor.predict_date(str(inference.latest_date().date()), top_n=10)
    assert len(out) == 10
    assert out["confidence"].between(0, 1).all()
