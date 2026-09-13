"""The lookup layer: what happens when a person types a word into the search box.

Search ranking is the part of the serving layer with no obviously-correct
answer, so these tests pin the judgement calls rather than the mechanics: an
exact ticker wins, a whole word beats a word fragment, and liquidity breaks ties
inside a bucket but never jumps one.
"""
import pandas as pd
import pytest

from serve import universe


@pytest.fixture
def stub_universe(monkeypatch):
    """A small universe built around the collisions that actually bite.

    'gold' appears as a whole word in three funds and as a fragment of
    "Goldman"; GLD is the fund a person typing "gold" means, and it is
    deliberately not the alphabetically-first match.
    """
    md = pd.DataFrame(
        {
            "name": [
                "SPDR Gold Shares",
                "iShares Gold Trust",
                "Goldman Sachs Access Treasury 0-1 Year ETF",
                "Direxion Daily Gold Miners Bull 2X",
                "Invesco QQQ Trust",
                "Vanguard Total Stock Market ETF",
                "Delisted Gold Thing",
            ],
            "category_group": ["Commodities"] * 4 + ["Equities"] * 2 + ["Commodities"],
            "category": ["Precious Metals"] * 4 + ["Large Cap", "Large Cap", "Precious Metals"],
            "family": ["State Street", "BlackRock", "Goldman Sachs",
                       "Direxion", "Invesco", "Vanguard", "Defunct"],
            "exchange": ["PCX"] * 7,
            "is_leveraged": [False, False, False, True, False, False, False],
            "is_delisted": [False, False, False, False, False, False, True],
        },
        index=pd.Index(["GLD", "IAU", "GBIL", "NUGT", "QQQ", "VTI", "OLDG"], name="ticker"),
    )
    for col in universe.SEARCH_FIELDS:
        md[f"_{col}_lc"] = md[col].str.lower()

    # `facets` is lru_cached over whatever `load_universe` returned, so both
    # caches have to be dropped on the way in and on the way out -- otherwise a
    # test either sees the real metadata file or leaks the stub into one.
    real_load = universe.load_universe
    real_load.cache_clear()
    universe.facets.cache_clear()
    monkeypatch.setattr(universe, "load_universe", lambda: md)
    yield md
    real_load.cache_clear()
    universe.facets.cache_clear()


def tickers(hits):
    return [h["ticker"] for h in hits]


# --------------------------------------------------------------------------- #
#  ranking                                                                    #
# --------------------------------------------------------------------------- #

def test_an_exact_ticker_wins_outright(stub_universe):
    """Typing a symbol you already know has to land on it, whatever else matches."""
    assert tickers(universe.search("qqq"))[0] == "QQQ"


def test_ticker_match_is_case_insensitive(stub_universe):
    assert tickers(universe.search("GlD"))[0] == "GLD"


def test_a_whole_word_beats_the_same_letters_inside_a_longer_word(stub_universe):
    """"gold" must not surface Goldman Sachs' treasury fund above actual gold
    ETFs -- this is the single worst failure a fund search can have."""
    ranked = tickers(universe.search("gold"))
    assert ranked.index("GBIL") > ranked.index("GLD")
    assert ranked.index("GBIL") > ranked.index("IAU")


def test_a_half_typed_word_still_finds_the_fund(stub_universe):
    """Incremental typing can't fall off a cliff: "goldm" has no whole-word
    match anywhere, so the word-prefix bucket has to catch it."""
    assert "GBIL" in tickers(universe.search("goldm"))


def test_liquidity_breaks_ties_within_a_match_bucket(stub_universe):
    """GLD and IAU match "gold" equally well on text; the liquid one goes first."""
    liq = pd.Series({"IAU": 5e8, "GLD": 9e9, "NUGT": 1e6})
    assert tickers(universe.search("gold", liquidity=liq))[:2] == ["GLD", "IAU"]

    flipped = pd.Series({"IAU": 9e9, "GLD": 5e8, "NUGT": 1e6})
    assert tickers(universe.search("gold", liquidity=flipped))[:2] == ["IAU", "GLD"]


def test_liquidity_never_promotes_a_worse_text_match(stub_universe):
    """A hugely liquid fragment-match still loses to a whole-word match. Ranking
    is match quality first; liquidity only orders within a tier."""
    liq = pd.Series({"GBIL": 1e12, "GLD": 1.0, "IAU": 1.0, "NUGT": 1.0})
    ranked = tickers(universe.search("gold", liquidity=liq))
    assert ranked.index("GBIL") > ranked.index("GLD")


def test_search_matches_issuer_and_category_too(stub_universe):
    assert "VTI" in tickers(universe.search("vanguard"))
    assert set(tickers(universe.search("precious metals"))) >= {"GLD", "IAU"}


def test_delisted_funds_are_hidden_unless_asked_for(stub_universe):
    assert "OLDG" not in tickers(universe.search("gold"))
    assert "OLDG" in tickers(universe.search("gold", include_delisted=True))


def test_search_honours_the_limit_and_shrugs_at_empty_input(stub_universe):
    assert len(universe.search("gold", limit=2)) == 2
    assert universe.search("") == []
    assert universe.search("   ") == []
    assert universe.search("zzzznotafund") == []


def test_a_regex_metacharacter_is_matched_literally(stub_universe):
    """Query text goes into a regex, so an unescaped '(' would be a 500."""
    assert universe.search("s&p 500 (") == []
    assert universe.search(".*") == []


# --------------------------------------------------------------------------- #
#  facets                                                                     #
# --------------------------------------------------------------------------- #

def test_facets_count_funds_per_value(stub_universe):
    facets = universe.facets()
    assert set(facets) == set(universe.FACET_FIELDS)
    groups = {f["value"]: f["count"] for f in facets["category_group"]}
    assert groups == {"Commodities": 5, "Equities": 2}


def test_facets_are_ordered_by_count(stub_universe):
    counts = [f["count"] for f in universe.facets()["category"]]
    assert counts == sorted(counts, reverse=True)


def test_resolve_returns_one_fund_or_none(stub_universe):
    assert universe.resolve("gld")["name"] == "SPDR Gold Shares"
    assert universe.resolve("NOPE") is None


# --------------------------------------------------------------------------- #
#  filtering a scored frame                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def scored():
    return pd.DataFrame(
        {"score": [0.4, 0.1, -0.3, -0.5],
         "confidence": [0.8, 0.5, 0.6, 0.9],
         "prediction": ["outperform", "neutral", "underperform", "underperform"],
         "name": ["Alpha Fund", "Beta Fund", "Gamma Fund", "Delta Fund"],
         "category": ["Large Cap", "Large Cap", "Bonds", "Bonds"],
         "is_leveraged": [False, True, False, False]},
        index=pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker"))


def test_filters_stack_rather_than_replace(scored):
    out, _ = universe.apply_filters(
        scored, filters={"category": ["Bonds"]}, predictions=["underperform"], min_score=-0.4)
    assert list(out.index) == ["CCC"]


def test_text_filter_matches_ticker_or_name(scored):
    assert list(universe.apply_filters(scored, q="bbb")[0].index) == ["BBB"]
    assert list(universe.apply_filters(scored, q="gamma")[0].index) == ["CCC"]


def test_leveraged_exclusion(scored):
    out, _ = universe.apply_filters(scored, exclude_leveraged=True)
    assert "BBB" not in out.index


def test_a_filter_on_a_missing_column_is_reported_not_silently_applied(scored):
    """Dropping every row because the deployment lacks a metadata column looks
    identical to 'nothing matched'. The caller has to be able to tell them apart."""
    out, ignored = universe.apply_filters(scored.drop(columns=["category"]),
                                          filters={"category": ["Bonds"]})
    assert ignored == ["category"]
    assert len(out) == len(scored)


def test_empty_filter_values_are_a_no_op(scored):
    out, ignored = universe.apply_filters(scored, filters={"category": []}, predictions=[])
    assert len(out) == len(scored)
    assert ignored == []
