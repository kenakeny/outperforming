# Pipeline, serving and tests (2026-07-25)

What this round added, and the one substantive finding that came out of building it.

## The gap this closed

The repo had a lot of research and no system. Concretely, before this round:

- `data/processed/dataset_v3.parquet` — the file **every** model in `train_models.py`
  trains on — had **no builder script**. Its 44 features existed only as executed
  cells in `notebooks/03_features.ipynb`. Nothing could rebuild it from raw data.
- There were **zero automated tests**, in a codebase whose entire value rests on
  leak-free features and a correctly purged walk-forward. The leak audits in
  `notebooks/common.py` were good, but they only ran when someone opened a notebook.
- There was **no serving layer at all** — no way to get a prediction out of the
  repo without loading a 3.3M-row parquet in a REPL.

## What's there now

| Piece | File | Notes |
|---|---|---|
| ETL pipeline | `etl.py` | 5 stages (extract → features → sr → labels → assemble), freshness-skipping, manifest, `--status` |
| Inference | `inference.py` | one loading path shared by both surfaces; feature order read from the artifact, not hardcoded |
| REST API | `serve/api.py` | FastAPI: `/health` `/models` `/metrics` `/signals` `/tickers/{t}` |
| Dashboard | `serve/dashboard.py` | Streamlit over the same `inference.py` |
| Linear baseline | `train_models.py` | added `fit_logreg` so logreg / xgboost / catboost / lstm sit in **one** comparison table |
| Tests | `tests/` | 60 fast + 44 slow parity tests |

Support/resistance features already existed (`build_sr_features.py`); the pipeline
now owns that construction as a stage so there's one copy of it.

### Test coverage, by what it protects

- **Lookahead** (`test_features_no_lookahead.py`) — rebuilds all 44 features and all
  11 S/R features from a panel truncated at *t* and asserts every value at *t* is
  unchanged. Also asserts no feature correlates with the forward return on a random
  walk. This is the failure mode that would silently invalidate every number in
  `reports/`.
- **Label + purge** (`test_labels_and_purge.py`) — leave-one-out benchmark, tercile
  balance, thin-category NaN, no label in the last H days, and that
  `assert_no_overlap` *actually fires* on a one-day-too-late cut. An assertion that
  can never fail protects nothing.
- **ETL contracts** (`test_etl_pipeline.py`) — universe filtering, schema/column
  order, float32, stage ordering, freshness invalidation by a newer upstream.
- **Serving** (`test_serving.py`) — feature-order independence, hard failure on a
  missing feature (never zero-fill), and that every response carries the
  not-investment-advice disclaimer and the macro-F1 convention.

## Finding: rebuilding the ETL does not reproduce dataset_v3 exactly, and that's informative

`etl.py` was validated against the existing `dataset_v3.parquet` over all 3,332,264
overlapping rows. The 44 features split cleanly in two:

- **31 features reproduce bit-for-bit** (max abs diff 0.0): all returns, EMA/MACD,
  ADX/ATR, RSI, Bollinger, OBV, MFI, CMF, dollar volume, every delta feature. The
  port is faithful.
- **13 features do not**, with correlations from 0.907 to 0.999. Every single one is
  either a **cross-sectional rank**, a **category aggregate**, or a **market-wide
  mean** — `beat_rate_20d` (0.907), `beta_60d`, `peer_rank_20d`, `cs_rank_*`,
  `cat_ret_5d`, `cat_disp_5d`, `mkt_*`.

The split is not random: it's exactly the features that depend on *which funds are in
the panel* and *what category each fund is in*. Both come from `metadata.parquet`,
which is re-downloaded periodically, and financedatabase's categories are **not
point-in-time**. `beat_rate_20d` at 0.907 is the loudest signal that peer groups have
genuinely moved since `dataset_v3` was built.

Two consequences worth taking seriously:

1. **A metadata refresh silently changes 13 features and the label itself**, because
   the peer group the label ranks within is what moved. Metrics reported before and
   after a refresh are not strictly comparable, and nothing in the repo said so.
   The README already lists non-point-in-time categories as a known limitation —
   this quantifies it.
2. The remaining four differ for benign, *verified* numerical reasons rather than
   being waved through:
   - `willr_14` / `stoch_k` / `stoch_d` divide by a 14-day high-low range that goes
     near-zero on a flat window. Correlation > 0.999; compared on rank.
   - `vol_zscore` uses `rolling(252, min_periods=60)`, whose valid-observation count
     depends on the panel's exact date rows. 9,464 of 3.2M rows (0.3%) differ, across
     74 tickers — and **100% of those tickers have interior NaN gaps**, versus 4.7%
     of the universe. The test now asserts exactly that: a mismatch on any fund with
     continuous history is a porting bug and fails.

`tests/test_dataset_parity.py` encodes all of this, so the distinction stays visible
instead of being rediscovered.

## Reality check on the signal

Nothing here changes the conclusion in `SESSION_SUMMARY.md` and
`SR_NEWS_MODEL_RESULTS.md`: pooled macro-F1 is ~0.436 against a ~0.364 majority-class
baseline, direction accuracy on genuinely extreme funds tops out near 55–56%, and S/R
and news both came in flat-to-negative versus the v3-only baseline. The API returns a
weak cross-sectional *ranking* signal and says so in every response.

## Not done

- **No backtest with transaction costs.** Macro-F1 does not tell you whether the
  signal survives spreads and turnover. For a trading agent this is the load-bearing
  question, and it's still open.
- The LSTM is trained and scored but not served — `inference.py` handles the tree
  models and the logreg pipeline; the LSTM needs its sequence-window loader wired in.
- `etl.py --stages extract` delegates to `build_dataset.py`'s downloader and hasn't
  been run end-to-end against a live yfinance pull in this round.
