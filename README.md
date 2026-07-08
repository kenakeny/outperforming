# fund short-term outperformance — data & eda

predict whether an etf will **Outperform**, be **Neutral**, or **Underperform** its category
peers over the next 5 trading days, using only public data (yahoo finance + financedatabase).

right now this repo covers **data ingestion + eda**

## structure

```
config.yaml     # exchanges, history length, labels etc
notebooks/
  01_data_ingestion.ipynb   # financedatabase universe + yfinance prices -> data/raw/
  02_eda.ipynb              # explore + label -> data/processed/labels.parquet
data/
  raw/         prices.parquet, metadata.parquet (+ csv copies)
  processed/   labels.parquet
```
