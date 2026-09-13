# ETF outperformance model

Ranks US ETFs against their category peers over a 10-trading-day horizon.
Builds 81 features from raw OHLCV, trains a two-stage XGBoost and a
category-graph neural net, blends them, evaluates on a held-out year.

## Setup

```bash
pip install -r requirements.txt
```

## Get the data

Two files under `data/raw/`:

| file | size | what |
|---|---|---|
| `market_data.parquet` | 152 MB | wide OHLCV panel, MultiIndex columns `(field, ticker)`, DatetimeIndex rows |
| `metadata.parquet` | 62 KB | one row per ticker, needs a `category` column |

Either copy those from me, **or** generate your own (free, no API key, ~15 min):

```bash
python build_dataset.py
```

Note: generating your own pulls a *current* snapshot of the fund universe and
categories, so your numbers won't match mine exactly. Categories from
`financedatabase` are not point-in-time — a refresh shifts the peer groups, which
shifts both features and the label.

## Run

```bash
python blend_standalone.py                  # full: XGBoost + GNN + blend
python blend_standalone.py --no-gnn         # XGBoost only, no torch needed
python blend_standalone.py --data-dir path/ # if the parquets live elsewhere
python blend_standalone.py --cpu            # no GPU
```

~20 min on a GPU. Writes `blend_results.json`.

## What to expect

`reference_results.json` has my run (2026 holdout, n=178,560):

| model | rank IC | dir. acc on extremes | macro-F1 |
|---|---|---|---|
| blend | **0.0852** | 0.5538 | 0.3801 |
| CatGNN alone | 0.0700 | 0.5427 | 0.3674 |
| XGBoost alone | 0.0257 | — | — |

## Reading the numbers

**rank IC is the metric that matters.** It's the daily Spearman correlation
between the model's score and the realized peer-relative return. macro-F1 is
reported for reference, but the two disagree — a model can classify well and rank
badly, and XGBoost here is exactly that case (macro-F1 ~0.50, rank IC 0.026).

**The macro-F1 in this script is not comparable to macro-F1 elsewhere.** This one
forces equal per-day terciles, so its baseline is 0.333. An argmax classifier on
the same data reports ~0.50 for the same skill level. Always check which
convention a number uses before comparing.

**Direction is the weak point.** ~55% accuracy on genuinely extreme funds. The
model detects *magnitude* well and *direction* poorly — it knows which funds will
move, not reliably which way. Don't build anything that assumes otherwise without
looking at the confusion matrix first.

## The one thing not to break

`EMBARGO` must equal `HORIZON`. The label at date *t* looks 10 days forward, so
training data must stop 10 trading days before the test window starts. A mismatch
here silently inflated every result in this project for months before it was
caught — the dataset carried a 25-day label while the code purged 5 days, leaking
20 days into every fold.

If you change `HORIZON`, change `EMBARGO` with it.
