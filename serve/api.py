"""serve/api.py -- FastAPI backend over the trained ETF outperformance models.

  uvicorn serve.api:app --reload      # docs at http://127.0.0.1:8000/docs

The read path is one shape repeated: `Predictor.scored_date()` produces the full
ranked frame for a date (cached), and every endpoint is a narrowing of it --
search, facet filters, sort, page. Nothing re-scores the universe to answer a
keystroke.

Endpoints
  GET  /health                     liveness + what data/models are loaded
  GET  /models                     registry + walk-forward scorecard
  GET  /metrics                    pooled walk-forward metrics per model
  GET  /dates                      scoreable trading dates (for the date picker)
  GET  /facets                     filter vocabulary: category groups, categories,
                                   families, exchanges + fund counts
  GET  /search?q=                  find a fund by ticker, name, family or category
  GET  /signals                    ranked signal for one day: search + filter +
                                   sort + paginate
  GET  /categories                 per-category roll-up for one day
  GET  /picks                      today's buy list, top N by score
  GET  /tickers/{ticker}           one fund: metadata, score history, prices
  POST /predict/csv                score an uploaded OHLCV CSV

  GET  /saudi/meta                 arms available, fund list, evaluation date range
  GET  /saudi/signals               ranked walk-forward record for one arm/date
  GET  /saudi/tickers/{ticker}     one Tadawul fund's scored history
  GET  /saudi/benchmark            model equity curve vs TASI/FALCOM, per arm

  The Saudi routes serve a *walk-forward evaluation record* (see serve/saudi.py),
  not a live model -- every response is flagged `is_realized_history: true` so a
  client can't present it as a same-footing signal with `/signals`.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import inference  # noqa: E402

from . import csv_input, saudi, universe

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(
    title="ETF Outperformance API",
    description="Peer-relative 5-day outperformance signal for US ETFs.",
    version="2.0.0",
)

# The React dev server runs on another port, so same-origin doesn't apply during
# development. Restricted to loopback -- in production the built frontend is
# served from this same app (see the StaticFiles mount at the bottom) and no
# cross-origin request is involved at all.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

#: Columns a caller may sort the signal table by. An arbitrary column name would
#: let a request sort by a raw feature, which is not a contract worth keeping.
SORTABLE = ("score", "confidence", "outperform", "neutral", "underperform",
            "ticker", "name", "category", "category_group", "family")

_predictors = {}


def get_predictor(name):
    """Load-once cache -- deserializing CatBoost per request is the slow path."""
    if name not in _predictors:
        try:
            _predictors[name] = inference.Predictor.load(name)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    return _predictors[name]


def _scored(model, date):
    """The full ranked frame for (model, date), or a 404 the client can act on."""
    predictor = get_predictor(model)
    date = date or str(inference.latest_date().date())
    try:
        return predictor.scored_date(date), date
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _check_facet_values(df, requested):
    """404 on a filter value no fund on this date carries.

    An unknown value is a client mistake worth reporting; a *combination* that
    happens to be empty is a legitimate answer and returns an empty page.
    """
    for field, values in requested.items():
        if not values or field not in df.columns:
            continue
        known = set(df[field].dropna().astype(str))
        unknown = [v for v in values if v not in known]
        if unknown:
            raise HTTPException(
                status_code=404,
                detail=f"no funds with {field} {unknown} on this date")


def _sort(df, sort, order):
    if sort not in SORTABLE:
        raise HTTPException(status_code=422,
                            detail=f"cannot sort by '{sort}'; choose one of {list(SORTABLE)}")
    ascending = order == "asc"
    if sort == "ticker":
        return df.sort_index(ascending=ascending)
    if sort not in df.columns:
        return df
    # na_position last in both directions: a fund with no name shouldn't take
    # the top of the table just because the sort flipped.
    return df.sort_values(sort, ascending=ascending, na_position="last")


def _records(df):
    """JSON-ready rows: NaN -> null, index -> a `ticker` field."""
    out = df.reset_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.where(out.notna(), None).to_dict(orient="records")


def _summary(df):
    """The headline numbers above the table, computed on the filtered set."""
    if df.empty:
        return {"n": 0, "mean_score": None, "mean_confidence": None,
                "outperform": 0, "neutral": 0, "underperform": 0}
    counts = df["prediction"].value_counts() if "prediction" in df.columns else {}
    return {"n": int(len(df)),
            "mean_score": float(df["score"].mean()),
            "mean_confidence": float(df["confidence"].mean()),
            "outperform": int(counts.get("outperform", 0)),
            "neutral": int(counts.get("neutral", 0)),
            "underperform": int(counts.get("underperform", 0))}


# --------------------------------------------------------------------------- #
#  service + reference data                                                   #
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    models = inference.available_models()
    try:
        latest = str(inference.latest_date().date())
        rows = len(inference.load_feature_panel())
    except FileNotFoundError:
        latest, rows = None, 0
    return {"status": "ok" if models and rows else "degraded",
            "models_available": sorted(models),
            "panel_rows": rows,
            "latest_date": latest,
            "frontend_built": WEB_DIST.exists()}


@app.get("/models")
def models():
    results = inference.load_results()
    return {"available": inference.available_models(),
            "walk_forward_results": results,
            "metric_convention": "macro-F1 over forced per-day terciles; "
                                 "majority-class baseline is ~0.364, not 0.333"}


@app.get("/metrics")
def metrics():
    results = inference.load_results()
    if not results:
        raise HTTPException(status_code=404, detail="no results.json -- run train_models.py")
    return results


@app.get("/dates")
def dates(limit: int = Query(750, ge=1, le=5000)):
    """The most recent scoreable trading dates, newest first."""
    try:
        available = inference.trading_dates()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="no feature panel -- run `python etl.py`")
    recent = available[-limit:][::-1]
    return {"latest": str(available.max().date()),
            "earliest": str(available.min().date()),
            "n_total": int(len(available)),
            "dates": [str(d.date()) for d in recent]}


@app.get("/facets")
def facets():
    """The filter vocabulary the UI builds its controls from."""
    return {"fields": list(universe.FACET_FIELDS),
            "facets": universe.facets(),
            "predictions": list(inference.CLASS_NAMES.values())}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="ticker, fund name, family or category"),
    limit: int = Query(20, ge=1, le=100),
    model: str = Query("catboost_sr", description="model used for the inline score"),
    date: str = Query(None, description="score date; defaults to latest"),
    with_scores: bool = Query(True, description="attach each hit's score on `date`"),
):
    """Fund lookup for the search box.

    Ranked so an exact ticker beats a name match: typing `SPY` lands on SPY, not
    on the first fund with "spy" somewhere in its prospectus name.
    """
    try:
        liq = inference.liquidity()
    except (FileNotFoundError, KeyError):
        liq = None

    hits = universe.search(q, limit=limit, liquidity=liq)
    if hits and with_scores:
        try:
            frame, date = _scored(model, date)
            cols = [c for c in ("score", "prediction", "confidence") if c in frame.columns]
            lookup = frame[cols].to_dict(orient="index")
        except HTTPException:
            lookup = {}
        for hit in hits:
            scores = lookup.get(hit["ticker"])
            # `scored` is not decoration: the metadata universe is wider than the
            # panel the model runs on, so a fund can be findable but unscoreable.
            # Saying so beats rendering an empty score cell.
            hit["scored"] = scores is not None
            hit.update(scores or {})
    return {"query": q, "n_hits": len(hits), "date": date, "results": hits}


# --------------------------------------------------------------------------- #
#  the signal table                                                           #
# --------------------------------------------------------------------------- #

@app.get("/signals")
def signals(
    date: str = Query(None, description="trading date (YYYY-MM-DD); defaults to latest"),
    model: str = Query("catboost_sr", description="model name from /models"),
    q: str = Query(None, description="free text: ticker, name, family or category"),
    category: list[str] = Query(None, description="peer category; repeatable"),
    category_group: list[str] = Query(None, description="broad asset group; repeatable"),
    family: list[str] = Query(None, description="issuer/fund family; repeatable"),
    exchange: list[str] = Query(None, description="listing exchange; repeatable"),
    prediction: list[str] = Query(None, description="outperform | neutral | underperform"),
    exclude_leveraged: bool = Query(False, description="drop leveraged/inverse products"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_score: float = Query(None, ge=-1.0, le=1.0),
    max_score: float = Query(None, ge=-1.0, le=1.0),
    sort: str = Query("score", description=f"one of {list(SORTABLE)}"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    top_n: int = Query(25, ge=1, le=500, description="page size"),
    offset: int = Query(0, ge=0, description="rows to skip, for paging"),
):
    """Ranked cross-sectional signal for one day, searched/filtered/sorted/paged.

    `score` is P(outperform) - P(underperform): a cross-sectional ranking signal
    over the fund's own category peers, not a return forecast.
    """
    frame, date = _scored(model, date)
    universe_scored = len(frame)

    requested = {"category": category, "category_group": category_group,
                 "family": family, "exchange": exchange}
    _check_facet_values(frame, requested)

    frame = frame[frame["confidence"] >= min_confidence]
    filtered, ignored = universe.apply_filters(
        frame, q=q, filters=requested, predictions=prediction,
        exclude_leveraged=exclude_leveraged, min_score=min_score, max_score=max_score)

    page = _sort(filtered, sort, order).iloc[offset:offset + top_n]

    return {"date": date, "model": model,
            "n_scored": len(filtered),          # rows matching the query
            "universe_scored": universe_scored,  # rows scored before filtering
            "offset": offset, "limit": top_n,
            "sort": sort, "order": order,
            "summary": _summary(filtered),
            "ignored_filters": ignored,
            "signals": _records(page)}


@app.get("/categories")
def categories(
    date: str = Query(None, description="trading date; defaults to latest"),
    model: str = Query("catboost_sr"),
    field: str = Query("category", description=f"group by one of {list(universe.FACET_FIELDS)}"),
    min_funds: int = Query(3, ge=1, description="drop groups thinner than this"),
):
    """Where the model is leaning, by peer group.

    Worth reading with the label in mind: the target ranks a fund *within* its
    category, so a category-wide mean score near zero is the expected case and a
    large one says the model disagrees with the peer group's own composition.
    """
    if field not in universe.FACET_FIELDS:
        raise HTTPException(status_code=422,
                            detail=f"cannot group by '{field}'; choose one of "
                                   f"{list(universe.FACET_FIELDS)}")
    frame, date = _scored(model, date)
    if field not in frame.columns:
        raise HTTPException(status_code=404, detail=f"metadata has no '{field}' column")

    grouped = frame.groupby(field)
    roll = pd.DataFrame({
        "n_funds": grouped.size(),
        "mean_score": grouped["score"].mean(),
        "mean_confidence": grouped["confidence"].mean(),
        "n_outperform": grouped["prediction"].apply(lambda s: int((s == "outperform").sum())),
        "n_underperform": grouped["prediction"].apply(lambda s: int((s == "underperform").sum())),
        "top_ticker": grouped["score"].idxmax(),
    })
    roll = roll[roll["n_funds"] >= min_funds].sort_values("mean_score", ascending=False)
    roll.index.name = field

    return {"date": date, "model": model, "field": field,
            "n_groups": len(roll),
            "groups": _records(roll)}


@app.get("/picks")
def picks(
    date: str = Query(None, description="trading date (YYYY-MM-DD); defaults to latest"),
    model: str = Query("catboost_sr", description="model name from /models"),
    n: int = Query(10, ge=1, le=100),
):
    """Straight buy list: the top N funds by score on this date, ranked, no probability table."""
    frame, date = _scored(model, date)
    df = frame.head(n).reset_index()
    cols = [c for c in ("ticker", "name", "category", "score", "confidence") if c in df.columns]
    return {"date": date, "model": model, "buy": df[cols].to_dict(orient="records")}


@app.get("/tickers/{ticker}")
def ticker_history(
    ticker: str,
    model: str = Query("catboost_sr"),
    days: int = Query(60, ge=1, le=1000),
):
    """One fund in full: what it is, how the model has scored it, what it did.

    The score series and the price series are returned separately rather than
    merged -- they share only their dates, and a client that merged them would
    be tempted to plot them on one pair of axes, which invents a relationship
    between a probability spread and a dollar price. They *are* clipped to a
    common date range, because the UI stacks them for visual comparison and the
    raw price pull runs later than the feature panel does.
    """
    predictor = get_predictor(model)
    panel = inference.load_feature_panel()
    ticker = ticker.upper()
    try:
        rows = panel.xs(ticker, level="ticker").tail(days)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticker '{ticker}'")

    scored = predictor.score(rows.dropna(subset=predictor.feature_names, how="all"))
    if "target" in rows.columns:
        scored["realized_target"] = rows["target"]
    scored.index.name = "date"
    history = scored.reset_index()
    history["date"] = history["date"].astype(str)

    prices = inference.load_prices()
    price_series = []
    if prices is not None and ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(scored):
            s = s.loc[s.index.isin(rows.index)]  # same trading days as the scores
        price_series = [{"date": str(d.date()), "close": float(v)} for d, v in s.tail(days).items()]

    latest = history.iloc[-1].to_dict() if len(history) else None
    return {"ticker": ticker,
            "model": model,
            "meta": universe.resolve(ticker),
            "latest": latest,
            "n_points": len(history),
            "history": history.where(history.notna(), None).to_dict(orient="records"),
            "prices": price_series}


# --------------------------------------------------------------------------- #
#  Saudi (Tadawul) transfer-learning record                                   #
# --------------------------------------------------------------------------- #

def _require_saudi():
    if not saudi.available():
        raise HTTPException(status_code=404,
                            detail="no reports/saudi_predictions.parquet -- run `python saudi.py`")


def _require_arm(arm):
    if arm not in saudi.arms():
        raise HTTPException(status_code=404,
                            detail=f"no predictions for arm '{arm}'; available: {saudi.arms()}")


@app.get("/saudi/meta")
def saudi_meta():
    """Arms available, the fund list, and the evaluated date range.

    There is no live Saudi model (see serve/saudi.py's docstring): this
    describes a fixed historical record, so the frontend can build an arm
    selector without a second round trip.
    """
    _require_saudi()
    preds = saudi.load_predictions()
    names = saudi.load_names()
    tickers = sorted(preds["ticker"].unique())
    dates = preds["date"].unique()
    dates = pd.Series(dates).sort_values(ascending=False)
    return {"arms": saudi.arms(),
            "horizon_days": saudi.HORIZON,
            "funds": [{"ticker": t, "name": names.get(t, t)} for t in tickers],
            "earliest": str(preds["date"].min().date()),
            "latest": str(preds["date"].max().date()),
            # newest-first, matching /dates -- the frontend's date picker is
            # shared between markets and expects this shape.
            "dates": [str(pd.Timestamp(d).date()) for d in dates],
            "is_realized_history": True}


@app.get("/saudi/signals")
def saudi_signals(
    arm: str = Query("saudi_only", description=f"one of {list(saudi.ARMS)}"),
    date: str = Query(None, description="evaluation date; defaults to latest"),
):
    """Every fund's score on one evaluation date, ranked -- with the outcome it
    was graded against alongside it, because for this record that outcome is
    already known and hiding it would make the row look like a live call.
    """
    _require_saudi()
    _require_arm(arm)
    try:
        ts, df = saudi.scored(arm, date)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"arm": arm, "date": str(ts.date()), "n": len(df),
            "is_realized_history": True,
            "signals": _records(df.set_index("ticker"))}


@app.get("/saudi/tickers/{ticker}")
def saudi_ticker_history(
    ticker: str,
    arm: str = Query("saudi_only", description=f"one of {list(saudi.ARMS)}"),
    days: int = Query(180, ge=1, le=1000),
):
    _require_saudi()
    _require_arm(arm)
    try:
        result = saudi.history(ticker, arm, days)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    rows = result["rows"]
    rows = rows.assign(date=rows["date"].astype(str))
    return {"ticker": ticker.upper(), "name": result["name"], "arm": arm,
            "is_realized_history": True,
            "history": rows.to_dict(orient="records")}


@app.get("/saudi/benchmark")
def saudi_benchmark(arm: str = Query("saudi_only", description=f"one of {list(saudi.ARMS)}")):
    """The model's equity curve against TASI and FALCOM over the same
    rebalance dates -- what saudi_vs_tasi_plot.py draws, as data.

    The TASI/FALCOM pull is a live yfinance call the first time it's asked for
    a given predictions-file snapshot; after that it's served from
    data/processed/transfer/saudi_benchmark_cache.json.
    """
    _require_saudi()
    _require_arm(arm)
    try:
        result = saudi.benchmark(arm)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"arm": arm, "horizon_days": saudi.HORIZON, "is_realized_history": True, **result}


# --------------------------------------------------------------------------- #
#  CSV upload                                                                 #
# --------------------------------------------------------------------------- #

@app.post("/predict/csv")
def predict_csv(
    file: UploadFile = File(..., description="OHLCV CSV: date, ticker, high, low, close, "
                                              "volume, and optionally category"),
    model: str = Query("catboost_sr", description="model name from /models"),
    latest_only: bool = Query(True, description="score only the most recent date per ticker "
                                                 "instead of every row in the upload"),
    top_n: int = Query(None, ge=1, le=2000, description="cap the number of rows returned"),
):
    """Score user-supplied OHLCV data instead of the pre-built historical panel.

    Technical features are computed straight from the uploaded prices. Cross-
    sectional / category features (peer_rank_20d, cat_ret_5d, beat_rate_20d, ...)
    are computed over whichever tickers are present in the upload: tickers that
    match the known universe (data/raw/metadata.parquet) use their real category
    as the peer group, unknown tickers are pooled into one catch-all peer group
    -- see serve/csv_input.py.
    """
    predictor = get_predictor(model)

    try:
        raw = csv_input.parse_ohlcv_csv(file.file)
        panel = csv_input.build_feature_panel(raw)
    except csv_input.CSVInputError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if latest_only:
        panel = panel.groupby(level="ticker").tail(1)

    scoreable = panel.dropna(subset=predictor.feature_names, how="all")
    if scoreable.empty:
        raise HTTPException(status_code=422,
                             detail=f"no row has enough trailing history to compute any "
                                    f"feature '{model}' needs -- upload more history per ticker")

    out = predictor.score(scoreable).sort_values("score", ascending=False).reset_index()
    out["date"] = out["date"].astype(str)

    return {"model": model,
            "n_input_rows": len(raw),
            "n_scored": len(out),
            "n_skipped_insufficient_history": len(panel) - len(scoreable),
            "predictions": (out.head(top_n) if top_n else out).to_dict(orient="records")}


# The built React app, if there is one. Mounted last so it can own "/" without
# shadowing any API route above; `html=True` makes client-side routing work.
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
