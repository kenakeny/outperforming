"""
inference.py -- load a trained model and score ETFs for a given date.

The serving layer (serve/api.py, serve/dashboard.py) and any future trading agent
go through this module, so "what the model saw" is defined in exactly one place.
Feature order comes from the saved model itself (CatBoost/XGBoost both record their
training feature names), not from a hand-maintained list that can silently drift
out of sync with the artifact.

  from inference import Predictor
  p = Predictor.load("catboost_sr")
  p.predict_date("2026-06-03", top_n=20)
"""
import collections
import functools
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"

CLASS_NAMES = {0: "underperform", 1: "neutral", 2: "outperform"}

# name -> (artifact file, loader kind). These are the artifacts train_models.py writes.
REGISTRY = {
    "catboost_sr": ("catboost_sr.cbm", "catboost"),
    "catboost_sr_news": ("catboost_sr_news.cbm", "catboost"),
    "xgboost_sr": ("xgboost_sr.ubj", "xgboost"),
    "xgboost_sr_news": ("xgboost_sr_news.ubj", "xgboost"),
    "logreg_sr": ("logreg_sr.joblib", "sklearn"),
    "logreg_sr_news": ("logreg_sr_news.joblib", "sklearn"),
}


def available_models():
    """Registry entries whose artifact actually exists on disk."""
    return {k: v[0] for k, v in REGISTRY.items() if (MODELS / v[0]).exists()}


@functools.lru_cache(maxsize=1)
def load_feature_panel():
    """The (date, ticker) feature panel to score from: dataset_v3 joined with S/R.

    Cached -- it's ~3.3M rows and the API would otherwise reload it per request.
    News columns are filled with 0.0 to match how train_models.py trained on them
    (news covers ~5% of rows; absent news means "no news", not "unknown").
    """
    d = pd.read_parquet(PROC / "dataset_v3.parquet")
    sr = PROC / "sr_features.parquet"
    if sr.exists():
        d = d.join(pd.read_parquet(sr), how="left")
    news = PROC / "news_features.parquet"
    if news.exists():
        nf = pd.read_parquet(news)
        d = d.join(nf, how="left")
        d[nf.columns] = d[nf.columns].fillna(0.0)
    return d.replace([np.inf, -np.inf], np.nan)


#: Metadata columns joined onto every prediction. `category` is the peer group
#: the label ranks within; the rest exist so the serving layer can search and
#: filter on them (serve/universe.py) without a second lookup.
META_COLS = ("name", "category_group", "category", "family", "exchange", "is_leveraged")


@functools.lru_cache(maxsize=1)
def load_metadata():
    md = pd.read_parquet(ROOT / "data" / "raw" / "metadata.parquet")
    return md[[c for c in META_COLS if c in md.columns]]


class Predictor:
    """A loaded model plus the feature contract it was trained with."""

    #: How many scored dates to keep in memory per predictor. A date is ~2k rows
    #: x ~12 columns, so this is a few MB total against re-running the model.
    DATE_CACHE_SIZE = 16

    def __init__(self, name, model, feature_names, kind):
        self.name = name
        self.model = model
        self.feature_names = list(feature_names)
        self.kind = kind
        self._date_cache = collections.OrderedDict()

    @classmethod
    def load(cls, name):
        if name not in REGISTRY:
            raise KeyError(f"unknown model '{name}'. available: {sorted(REGISTRY)}")
        filename, kind = REGISTRY[name]
        path = MODELS / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python train_models.py` first")

        if kind == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(str(path))
            names = model.feature_names_
        elif kind == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            names = model.get_booster().feature_names
        elif kind == "sklearn":
            import joblib
            model = joblib.load(path)
            # the pipeline's first step records the columns it was fitted on
            names = list(model[0].feature_names_in_)
        else:
            raise ValueError(f"unsupported model kind: {kind}")
        return cls(name, model, names, kind)

    def predict_frame(self, X):
        """Class probabilities for a feature frame, aligned to this model's columns.

        Missing columns are a hard error rather than a silent zero-fill: scoring a
        model against the wrong feature set produces plausible-looking numbers that
        mean nothing, which is worse than a traceback.
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"feature panel is missing {len(missing)} column(s) "
                             f"required by '{self.name}': {missing[:8]}")
        proba = self.model.predict_proba(X[self.feature_names])
        return pd.DataFrame(proba, index=X.index,
                            columns=[CLASS_NAMES[i] for i in range(proba.shape[1])])

    def score(self, X):
        """Class probabilities plus the derived score/prediction/confidence
        columns every caller needs -- predict_date, the CSV-upload endpoint, and
        the dashboard's upload tab all build on this instead of each computing
        it themselves.

        `score` = P(outperform) - P(underperform): a single signed number in
        [-1, 1] that a downstream agent can sort or threshold on. This is a
        cross-sectional ranking signal, not a return forecast.
        """
        proba = self.predict_frame(X)
        out = proba.copy()
        out["score"] = out["outperform"] - out["underperform"]
        out["prediction"] = proba.values.argmax(axis=1)
        out["prediction"] = out["prediction"].map(CLASS_NAMES)
        out["confidence"] = proba.max(axis=1)
        return out

    def scored_date(self, date):
        """Every fund scored on `date`, ranked, with metadata joined -- cached.

        This is the frame the whole serving layer narrows down: search, facet
        filters, sorting and pagination are all views over it. Caching it here
        means a user typing in the search box or clicking through pages costs
        a dataframe slice, not 2,000 model evaluations per keystroke.

        Returns a copy, so a caller mutating the result can't poison the cache.
        """
        ts = pd.Timestamp(date)
        if ts in self._date_cache:
            self._date_cache.move_to_end(ts)
            return self._date_cache[ts].copy()

        panel = load_feature_panel()
        try:
            rows = panel.xs(ts, level="date")
        except KeyError:
            raise KeyError(f"no data for {ts.date()}; latest available is {latest_date().date()}")

        rows = rows.dropna(subset=self.feature_names, how="all")
        out = self.score(rows).join(load_metadata(), how="left")
        out = out.sort_values("score", ascending=False)
        out.index.name = "ticker"

        self._date_cache[ts] = out
        while len(self._date_cache) > self.DATE_CACHE_SIZE:
            self._date_cache.popitem(last=False)
        return out.copy()

    def predict_date(self, date, top_n=None, min_probability=0.0):
        """Score every fund on `date` and return them ranked by conviction."""
        out = self.scored_date(date)
        out = out[out["confidence"] >= min_probability]
        return out.head(top_n) if top_n else out


def latest_date():
    return load_feature_panel().index.get_level_values("date").max()


def trading_dates():
    """Every date the panel can be scored on, ascending.

    The UI's date picker needs these: the panel skips weekends and holidays, so
    offering a free calendar just lets people pick days that 404.
    """
    return load_feature_panel().index.get_level_values("date").unique().sort_values()


@functools.lru_cache(maxsize=1)
def liquidity():
    """Each fund's most recent 20-day dollar volume, or None if unavailable.

    The serving layer uses this to rank search hits: for a query like "gold",
    the fund someone means is almost always the liquid one, not the triple-
    levered ETN that happens to share the word. Not every fund in the metadata
    universe reaches the panel, so this is a partial index by construction.
    """
    panel = load_feature_panel()
    if "dollar_vol_20d" not in panel.columns:
        return None
    return panel["dollar_vol_20d"].dropna().groupby(level="ticker").last()


@functools.lru_cache(maxsize=1)
def load_prices():
    """Wide close-price frame (date x ticker), or None if it hasn't been pulled.

    Only the serving layer uses this -- models never see raw price levels, but
    a person looking at a fund wants to see what the price did.
    """
    path = ROOT / "data" / "raw" / "prices.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_results():
    """The walk-forward scorecard train_models.py wrote, if present."""
    p = MODELS / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}
