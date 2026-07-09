# fund short-term outperformance — data & eda

predict whether an etf will **Outperform**, be **Neutral**, or **Underperform** its category
peers over the next 5 trading days, using only public data (yahoo finance + financedatabase).

right now this repo covers **data ingestion + eda + feature engineering**

## structure

```
config.yaml     # exchanges, history length, features, labels etc
notebooks/
  01_data_ingestion.ipynb   # financedatabase universe + yfinance prices -> data/raw/
  02_eda.ipynb              # explore + label -> data/processed/labels.parquet
  03_features.ipynb         # technical features + modeling dataset -> data/processed/
data/
  raw/         prices.parquet, metadata.parquet (+ csv copies)
  processed/   labels.parquet, features.parquet, dataset.parquet
reports/       eda dashboard pngs (regen: python reports/make_figures.py)
```