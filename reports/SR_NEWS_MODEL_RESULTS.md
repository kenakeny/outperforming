# Support/Resistance + News models — results (2026-07-23)

> ## ⚠️ THE RESULTS TABLE BELOW IS INVALID — do not quote these numbers
>
> Every metric in this report was produced from `data/processed/dataset_v3.parquet`,
> whose label was later found to be `close.shift(-25)/close.shift(-5) - 1` — a
> 20-trading-day forward return starting 5 days ahead, with a label window spanning
> `t+5 … t+25`. `train_models.py` purged only 5 trading days, leaving **20 trading
> days of label leakage in every walk-forward fold**.
>
> See [LABEL_HORIZON_BUG.md](LABEL_HORIZON_BUG.md) for the evidence and the fix.
> The dataset has been rebuilt with a true 5-day label and the models retrained;
> current numbers are in `models/results.md`.
>
> The *qualitative* conclusion below — that S/R and news add nothing over the v3-only
> baseline — was re-confirmed on the corrected label, but the figures themselves were
> measured against the wrong target and are not comparable to anything.

## What was built
- **`build_sr_features.py`** → `data/processed/sr_features.parquet` (11 features,
  4.96M rows). Swing-based support/resistance from OHLC (`market_data.parquet`),
  all `shift(1)`-trailing (no lookahead): `dist_res_{20,60}`, `dist_sup_{20,60}`,
  `range_pos_{20,60}`, `sr_width_20`, `res_touch_20`, `sup_touch_20`,
  `broke_res_20`, `broke_sup_20`.
- **`train_models.py`** → trains XGBoost / CatBoost / LSTM on `dataset_v3` (44
  feats) ± S/R (11) ± news (4), purged walk-forward (horizon 5, test 2019–26).
  Saves every model to `models/`.

## Results — pooled macro-F1 (2019–2026)
| Model | features | macro-F1 |
|---|---|---|
| **CatBoost** | **base (v3 only)** | **0.4360** |
| CatBoost | +sr+news | 0.4355 |
| CatBoost | +sr | 0.4351 |
| XGBoost | base (v3 only) | 0.4336 |
| XGBoost | +sr | 0.4329 |
| XGBoost | +sr+news | 0.4329 |
| LSTM | +sr+news | 0.4263 |

Majority-class baseline ≈ 0.364.

## Key finding
**Neither support/resistance nor news improves the models** — both are flat-to-
slightly-negative vs the v3-only baseline. The v3 price/volume/rank features
already capture the signal S/R encodes (distance-to-high, range position, etc.),
so the new S/R columns are largely redundant; news is too sparse (5% coverage).
Base CatBoost (v3 only) remains the best model.

## Saved models (`models/`)
`xgboost_sr.ubj`, `xgboost_sr_news.ubj`, `catboost_sr.cbm`,
`catboost_sr_news.cbm`, `lstm_sr_news.pt` (+ `_meta.json`), `results.json/md`.
Each tree file is the last walk-forward fold's fitted model; the LSTM meta holds
the feature list, seq length, and standardization stats for inference.
