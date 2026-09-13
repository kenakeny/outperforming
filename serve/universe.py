"""serve/universe.py -- the lookup layer: how a human finds a fund.

The model side of this repo indexes everything by ticker, but nobody remembers
that IHAK is the cybersecurity ETF. This module is the translation layer between
"what a person types" and "which rows of the panel to score": free-text search
over ticker/name/family, and the facet vocabulary (category, category group,
family, exchange) the UI turns into filter controls.

Search and filtering both work off `data/raw/metadata.parquet` -- the same file
`inference.load_metadata` joins onto predictions -- so a fund found in the search
box is guaranteed to be the fund scored in the table.

NOTE on categories: financedatabase's categories are not point-in-time and the
file is periodically re-downloaded. A refresh moves funds between peer groups,
which changes 13 features and the label (see etl.py). Search results are
therefore "as of the current metadata snapshot", not as of the scored date.
"""
import functools
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
META_PATH = ROOT / "data" / "raw" / "metadata.parquet"

# The dimensions the UI offers as filter controls. Order is the order they're
# presented in: broadest bucket first, narrowest last.
FACET_FIELDS = ("category_group", "category", "family", "exchange")

# Text columns free-text search looks at, best-signal first.
SEARCH_FIELDS = ("name", "family", "category")

# Rank buckets for a query match, lower = better. Sorting on buckets rather than
# on a similarity score is what keeps "TQQQ" pinned to the top when someone types
# TQQQ, and what keeps a *word* match ("SPDR Gold Shares") above a match buried
# inside a longer word ("Goldman Sachs") -- searching "gold" for gold ETFs and
# getting the issuer's bond funds is the failure mode this ordering exists for.
_EXACT_TICKER = 0
_TICKER_PREFIX = 1
_NAME_WORD = 2         # query is a whole word in the name ("Gold" in "SPDR Gold Shares")
_NAME_WORD_PREFIX = 3  # query starts a longer word ("gold" in "Goldman") -- also what
                       # a half-typed query looks like, so it stays above substrings
_TICKER_SUBSTR = 4
_NAME_SUBSTR = 5       # query appears mid-word in the name
_OTHER_WORD = 6        # whole word / word start in the family or category
_OTHER_SUBSTR = 7


@functools.lru_cache(maxsize=1)
def load_universe():
    """Fund metadata indexed by ticker, with lowercased text for matching.

    Cached: it's ~2k rows read once, and every search/facet call needs it.
    """
    if not META_PATH.exists():
        return pd.DataFrame(index=pd.Index([], name="ticker"))
    md = pd.read_parquet(META_PATH)
    md.index = md.index.astype(str).str.upper()
    md.index.name = "ticker"
    for col in SEARCH_FIELDS:
        if col in md.columns:
            md[f"_{col}_lc"] = md[col].fillna("").astype(str).str.lower()
    return md


_BOUNDARY = "[^a-z0-9]"


def _whole_word(series, ql):
    """True where `ql` is a complete word: 'gold' in "SPDR Gold Shares", not "Goldman"."""
    return series.str.contains(rf"(?:^|{_BOUNDARY}){re.escape(ql)}(?:{_BOUNDARY}|$)", regex=True)


def _word_start(series, ql):
    """True where `ql` begins a word -- start of string or after a non-alphanumeric."""
    return series.str.contains(rf"(?:^|{_BOUNDARY}){re.escape(ql)}", regex=True)


def _rank(md, q):
    """Sort keys per ticker for query `q`: (bucket, match position, name length).

    Rows that matched nothing get a NaN bucket. Position and length are the
    tiebreakers inside a bucket, so a fund whose name *leads* with the query
    beats one that mentions it in passing, and the shorter of two equally-placed
    matches wins ("SPDR Gold Shares" over "iShares Gold Strategy ETF Trust").
    """
    ql = q.strip().lower()
    tickers = md.index.to_series().str.lower()
    name = md["_name_lc"] if "_name_lc" in md.columns else pd.Series("", index=md.index)
    bucket = pd.Series(float("nan"), index=md.index)

    def mark(mask, value):
        bucket.loc[mask.fillna(False) & bucket.isna()] = value

    mark(tickers.eq(ql), _EXACT_TICKER)
    mark(tickers.str.startswith(ql), _TICKER_PREFIX)
    mark(_whole_word(name, ql), _NAME_WORD)
    mark(_word_start(name, ql), _NAME_WORD_PREFIX)
    mark(tickers.str.contains(ql, regex=False), _TICKER_SUBSTR)
    mark(name.str.contains(ql, regex=False), _NAME_SUBSTR)
    for col in SEARCH_FIELDS[1:]:
        lc = f"_{col}_lc"
        if lc in md.columns:
            mark(_word_start(md[lc], ql), _OTHER_WORD)
    for col in SEARCH_FIELDS[1:]:
        lc = f"_{col}_lc"
        if lc in md.columns:
            mark(md[lc].str.contains(ql, regex=False), _OTHER_SUBSTR)

    position = name.str.find(ql).replace(-1, 999)
    return pd.DataFrame({"_bucket": bucket, "_pos": position, "_len": name.str.len()})


def search(q, limit=20, include_delisted=False, liquidity=None):
    """Funds matching `q`, best match first.

    Matches ticker, name, family and category. An exact ticker hit always
    outranks a name hit, so typing a symbol you know lands on it directly.

    `liquidity` (a ticker -> dollar-volume Series) breaks ties *within* a match
    bucket. It matters: "gold" matches a dozen funds equally well on text alone,
    and the one the user means is the one people actually trade. Match quality
    still comes first -- a liquid fund never jumps a bucket.
    """
    md = load_universe()
    if md.empty or not q or not q.strip():
        return []

    if not include_delisted and "is_delisted" in md.columns:
        md = md[~md["is_delisted"].fillna(False).astype(bool)]

    keys = _rank(md, q)
    matched = keys["_bucket"].notna()
    hits = md[matched].join(keys[matched])

    if liquidity is not None:
        hits["_liq"] = -liquidity.reindex(hits.index).fillna(0.0)
    else:
        hits["_liq"] = 0.0

    hits = hits.sort_values(["_bucket", "_liq", "_pos", "_len", "ticker"], kind="stable")
    return _records(hits.head(limit))


def _records(md):
    """Metadata rows as JSON-ready dicts, dropping the internal match columns."""
    cols = [c for c in ("name", "category_group", "category", "family",
                        "exchange", "is_leveraged") if c in md.columns]
    out = md[cols].reset_index()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(bool)
    return out.where(out.notna(), None).to_dict(orient="records")


@functools.lru_cache(maxsize=1)
def facets():
    """Every filterable value with how many funds carry it.

    Counts come from metadata rather than from a scored date so the filter
    controls are stable as the user moves the date around -- a control that
    reshuffles under you is worse than one whose counts are approximate.
    """
    md = load_universe()
    out = {}
    for field in FACET_FIELDS:
        if field not in md.columns:
            out[field] = []
            continue
        counts = md[field].fillna("Uncategorized").value_counts()
        out[field] = [{"value": str(v), "count": int(n)} for v, n in counts.items()]
    return out


def resolve(ticker):
    """One fund's metadata, or None if it isn't in the universe."""
    md = load_universe()
    ticker = str(ticker).upper()
    if ticker not in md.index:
        return None
    return _records(md.loc[[ticker]])[0]


def apply_filters(df, q=None, filters=None, predictions=None,
                  exclude_leveraged=False, min_score=None, max_score=None):
    """Narrow a scored frame (index=ticker) by everything the UI can ask for.

    Returns `(filtered, ignored)`. A filter naming a column this frame doesn't
    carry is reported in `ignored` rather than silently dropping every row --
    a filter that can't be honoured has to be visible, not invisible.
    """
    ignored = []
    out = df

    for field, values in (filters or {}).items():
        values = [v for v in (values or []) if v]
        if not values:
            continue
        if field not in out.columns:
            ignored.append(field)
            continue
        out = out[out[field].isin(values)]

    if predictions:
        if "prediction" in out.columns:
            out = out[out["prediction"].isin(predictions)]
        else:
            ignored.append("prediction")

    if exclude_leveraged:
        if "is_leveraged" in out.columns:
            out = out[~out["is_leveraged"].fillna(False).astype(bool)]
        else:
            ignored.append("is_leveraged")

    if min_score is not None:
        out = out[out["score"] >= min_score]
    if max_score is not None:
        out = out[out["score"] <= max_score]

    if q and q.strip():
        out = _text_filter(out, q)

    return out, ignored


def _text_filter(df, q):
    """Rows whose ticker or any searchable text column contains `q`.

    Falls back to a ticker-only match when the frame carries no text columns,
    which is what a bare feature panel looks like.
    """
    ql = q.strip().lower()
    mask = df.index.to_series().str.lower().str.contains(ql, regex=False)
    for col in SEARCH_FIELDS:
        if col in df.columns:
            mask |= df[col].fillna("").astype(str).str.lower().str.contains(ql, regex=False)
    return df[mask]
