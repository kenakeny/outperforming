"""
serve/dashboard.py -- Streamlit front end over the same inference module the API uses.

  streamlit run serve/dashboard.py

Reads `inference` (and, for the upload tab, `csv_input`) directly rather than
calling the FastAPI service, so the dashboard works standalone; every surface
-- API, dashboard, CLI -- shares the same feature/scoring path so they can't
disagree about what the model saw.
"""
import json
import pathlib
import sys

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKTEST_DIR = ROOT / "reports" / "us_backtest"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serve"))

import csv_input  # noqa: E402
import inference  # noqa: E402

st.set_page_config(page_title="ETF Outperformance", layout="wide")


@st.cache_resource
def get_predictor(name):
    return inference.Predictor.load(name)


@st.cache_data
def get_panel_info():
    return str(inference.latest_date().date()), len(inference.load_feature_panel())


@st.cache_data
def get_spy_data():
    path = BACKTEST_DIR / "spy_data.json"
    return json.loads(path.read_text()) if path.exists() else None


st.title("ETF peer-relative outperformance")
st.caption("5-trading-day horizon, ranked within each fund's category.")

available = inference.available_models()
if not available:
    st.error("No trained models in `models/`. Run `python train_models.py` first.")
    st.stop()

latest, n_rows = get_panel_info()

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Model", sorted(available))
    date = st.date_input("Date", value=pd.Timestamp(latest).date())
    top_n = st.slider("Top N", 5, 100, 25)
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    st.caption(f"Panel: {n_rows:,} rows · latest {latest}")

predictor = get_predictor(model_name)

tab_buy, tab_upload, tab_backtest = st.tabs(["What to buy", "Upload CSV", "Backtest vs SPY"])

with tab_buy:
    try:
        scored = predictor.predict_date(str(date), min_probability=min_conf)
    except KeyError as e:
        st.warning(str(e))
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Funds scored", f"{len(scored):,}")
    c2.metric("Predicted outperform", int((scored["prediction"] == "outperform").sum()))
    c3.metric("Mean confidence", f"{scored['confidence'].mean():.3f}")

    display_cols = [c for c in ["name", "category", "prediction", "score", "confidence",
                                "outperform", "neutral", "underperform"] if c in scored.columns]

    left, right = st.columns(2)
    with left:
        st.subheader(f"Buy — top {top_n}")
        st.dataframe(scored.head(top_n)[display_cols], use_container_width=True)
    with right:
        st.subheader(f"Avoid — bottom {top_n}")
        st.dataframe(scored.tail(top_n)[display_cols].iloc[::-1], use_container_width=True)

    st.subheader("Score distribution")
    st.bar_chart(scored["score"].value_counts(bins=40, sort=False).rename("funds"))

    if "category" in scored.columns:
        st.subheader("Mean score by category")
        by_cat = (scored.groupby("category")["score"]
                  .agg(["mean", "count"]).sort_values("mean", ascending=False))
        st.dataframe(by_cat[by_cat["count"] >= 3], use_container_width=True)

    with st.expander("Walk-forward scorecard"):
        results = inference.load_results()
        if results:
            st.dataframe(pd.DataFrame(results).T)
            st.caption("macro-F1 over forced per-day terciles. Majority-class baseline ≈ 0.364.")
        else:
            st.info("No `models/results.json` yet — run `python train_models.py`.")

with tab_upload:
    st.subheader("Score your own OHLCV data")
    st.caption("CSV columns: `date, ticker, high, low, close, volume` (optional "
               "`category`; `symbol` / `adj close` also accepted as aliases). "
               "Peer/category features are computed across whatever tickers you "
               "upload — tickers matching the known universe use their real "
               "category as peer group, unrecognized ones are pooled into one "
               "catch-all peer group.")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    col_a, col_b = st.columns(2)
    latest_only = col_a.checkbox("Score latest date per ticker only", value=True)
    csv_top_n = col_b.slider("Rows to show", 5, 200, 25, key="csv_top_n")

    if uploaded is None:
        st.info("Upload a CSV to see predictions here.")
    else:
        try:
            raw = csv_input.parse_ohlcv_csv(uploaded)
            panel = csv_input.build_feature_panel(raw)
        except csv_input.CSVInputError as e:
            st.error(str(e))
            st.stop()

        if latest_only:
            panel = panel.groupby(level="ticker").tail(1)

        scoreable = panel.dropna(subset=predictor.feature_names, how="all")
        if scoreable.empty:
            st.warning(f"No row has enough trailing history to compute any feature "
                       f"'{model_name}' needs — upload more history per ticker.")
            st.stop()

        result = predictor.score(scoreable).sort_values("score", ascending=False).reset_index()
        n_skipped = len(panel) - len(scoreable)
        msg = f"Scored {len(result)} row(s) from {raw['ticker'].nunique()} ticker(s)"
        if n_skipped:
            msg += f" · {n_skipped} skipped for insufficient history"
        st.success(msg)
        st.dataframe(result.head(csv_top_n), use_container_width=True)

with tab_backtest:
    st.subheader("Model vs. SPY, 2019–2026 walk-forward")

    spy = get_spy_data()
    if spy is None:
        st.info(f"No backtest data at `{BACKTEST_DIR / 'spy_data.json'}` — "
                f"run `python reports/us_backtest/plot_backtest.py` first.")
    else:
        rows = []
        for label, key in [("screened (≥$50M ADV)", "screened"), ("unscreened", "unscreened")]:
            s = spy[key]
            rows.append({"portfolio": label,
                        "ann. return": f"{s['port_ann']:.1%}",
                        "SPY ann. return": f"{s['spy_ann']:.1%}",
                        "ann. alpha": f"{s['alpha_ann']:.1%}",
                        "information ratio": f"{s['ir']:.2f}",
                        "hit rate": f"{s['hit']:.1%}",
                        "beta to SPY": f"{s['beta']:.2f}"})
        st.dataframe(pd.DataFrame(rows).set_index("portfolio"), use_container_width=True)

        equity_png = BACKTEST_DIR / "backtest_equity.png"
        overview_png = BACKTEST_DIR / "backtest_overview.png"
        if equity_png.exists():
            st.image(str(equity_png), use_container_width=True)
        if overview_png.exists():
            with st.expander("Full backtest breakdown"):
                st.image(str(overview_png), use_container_width=True)
