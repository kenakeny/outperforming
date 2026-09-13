# FinBERT news-sentiment models — results (2026-07-23)

## Pipeline
1. `fetch_etf_news.py` (Polygon) + `fetch_etf_news_finnhub.py` → 349k article rows
   (192,817 Polygon / 156,245 Finnhub), 64,650 unique after dedup by id.
2. `score_sentiment.py` — FinBERT (`ProsusAI/finbert`, safetensors) on GPU,
   signed score = P(pos) − P(neg). Assigned to the *next* trading day after
   publication (no lookahead). → `data/processed/news_features.parquet`
   (210,446 date×ticker rows, 1,844 ETFs): `news_sent_1d/5d`, `news_n_1d/5d`.
3. `train_news_models.py` — `dataset_v3` (44 feats) ± 4 news feats, purged
   walk-forward (horizon 5), test years 2019–2026. Label = 3-class threshold
   tercile (0 under / 1 neutral / 2 outperform peers).

## Results — pooled macro-F1 (accuracy)
| Model            | macro-F1 | accuracy |
|------------------|----------|----------|
| CatBoost base    | **0.4360** | 0.4318 |
| CatBoost +news   | 0.4357   | 0.4316 |
| XGBoost +news    | 0.4339   | 0.4294 |
| XGBoost base     | 0.4336   | 0.4290 |
| LSTM +news (20d) | 0.4284   | 0.4266 |

Majority-class baseline ≈ 0.364 accuracy → all models carry real signal.

## Key finding
**News sentiment adds essentially nothing** (±0.0003 macro-F1). Only **5.16%**
of (date,ticker) rows have any same-day news, so the sentiment feature is 0 for
95% of the panel — too sparse to move a cross-sectional model. To make news
useful it would need to be restricted to the liquid, well-covered subset (and
likely modeled there separately), not bolted onto the full 1,975-ETF universe.
