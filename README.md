# Overperforming

An end-to-end research application for ranking exchange-traded funds by their expected **peer-relative return over the next five trading days**.

The project combines a reproducible market-data pipeline, leakage-aware walk-forward evaluation, six trained classification models, a FastAPI service, and a React screener. It predicts one of three classes—`outperform`, `neutral`, or `underperform`—relative to funds in the same category. It does **not** predict an absolute price target or guarantee positive returns.

## What is included

- A staged ETF data pipeline: extraction, cleaning, feature engineering, support/resistance features, labels, and dataset assembly.
- Logistic regression, XGBoost, CatBoost, and LSTM research paths.
- Purged walk-forward validation with explicit label-window overlap checks.
- A FastAPI API for model discovery, ranked signals, search, facets, categories, ticker history, and CSV uploads.
- A React/Vite interface with US-market and Saudi-market views.
- A historical Saudi transfer-learning evaluation. This view is realized backtest history, not a live Saudi trading signal.
- Unit and integration tests covering pipeline contracts, lookahead protection, serving, search/filter behavior, CSV ingestion, and Saudi endpoints.

## Architecture

```text
Yahoo Finance + fund metadata
            |
            v
   staged ETL pipeline  --->  feature panel + peer-relative labels
            |                              |
            v                              v
    walk-forward training  ----------> trained model artifacts
                                             |
                                             v
                                      FastAPI service
                                             |
                                             v
                                      React screener
```

All serving paths use `inference.py`, which reads feature names from each trained artifact and rejects incompatible feature panels rather than silently filling missing inputs.

## Quick start

### Prerequisites

- Python 3.10+
- Node.js 18+

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install and build the frontend:

```bash
npm --prefix web install
npm --prefix web run build
```

Build the local dataset before using the live US screener:

```bash
python etl.py --status
python etl.py --stages all
```

Then start the application:

```bash
uvicorn serve.api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Downloaded data is intentionally excluded from Git. A fresh clone therefore needs the ETL step before date-based US endpoints can return predictions. The CSV-upload endpoint can score a compatible OHLCV file without publishing the local research dataset.

### Frontend development

Run the API on port 8000, then start Vite in a second terminal:

```bash
npm --prefix web run dev
```

Vite serves the frontend at [http://localhost:5173](http://localhost:5173) and proxies API calls to FastAPI.

## Data and training

The main pipeline is controlled by `config.yaml`:

```bash
python etl.py --status       # inspect stage freshness
python etl.py --stages all   # rebuild the complete dataset
python train_models.py       # train the 5-day model family
pytest                       # run the fast test suite
pytest -m slow               # include full-panel parity checks
```

Optional news features use Polygon or Finnhub. Copy the example environment file and add only the key you need:

```bash
cp .env.example .env
python -m scripts.data.fetch_etf_news --dry-run
python -m scripts.data.fetch_etf_news_finnhub --dry-run
```

Never commit `.env`; it is ignored by Git.

## Evaluation methodology

The target is a forward five-trading-day return ranked within each date and ETF category. Because adjacent labels overlap, ordinary random splits would overstate performance. The repository uses:

- expanding-window, time-ordered folds;
- a trading-day purge between training and evaluation windows;
- assertions that no training label window reaches the test period;
- feature truncation tests that recompute features using only information available at that timestamp;
- shuffled-label and planted-leak checks;
- an additional non-overlapping evaluation sampled every fifth trading day.

The project presentation reports these headline held-out results:

| Predictor | Accuracy | ROC-AUC |
| --- | ---: | ---: |
| ETF performance predictor | 63% | 75% |
| ETF volatility predictor | 77% | 86% |

The performance predictor classifies peer-relative direction. The volatility predictor classifies high- and low-volatility bands using the same underlying data pipeline. These presentation figures are historical research results, not expected live returns.

## API overview

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service, model, and data readiness |
| `GET` | `/models`, `/metrics` | Model registry and stored evaluation metrics |
| `GET` | `/dates` | Available scoring dates |
| `GET` | `/search`, `/facets` | Fund lookup and filter vocabulary |
| `GET` | `/signals`, `/categories`, `/picks` | Ranked and aggregated US signals |
| `GET` | `/tickers/{ticker}` | Fund metadata, scores, and available price history |
| `POST` | `/predict/csv` | Score uploaded OHLCV data |
| `GET` | `/saudi/*` | Historical Saudi evaluation records and benchmarks |

The Saudi endpoints include `is_realized_history: true` in their responses so clients can distinguish the evaluation record from live US scoring.

## Repository layout

```text
config.yaml             pipeline, label, split, and path configuration
etl.py                  staged data pipeline
core.py                 shared feature and label implementation
inference.py            single model-loading and scoring path
train_models.py         primary model training and walk-forward evaluation
saudi.py                Saudi transfer-learning experiment
scripts/
  data/                  dataset builders and optional news ingestion
  training/              alternate models, tuning, and ablations
  evaluation/            backtests, plots, and overfit audits
  automation/            notebook and experiment runners
serve/
  api.py                FastAPI application
  universe.py           search, facets, and filters
  csv_input.py          uploaded OHLCV validation and feature construction
  saudi.py              historical Saudi results adapter
web/
  src/                  React application and components
tests/                  pipeline, leakage, API, and UI-contract tests
notebooks/              ingestion, EDA, features, and model research
experiments/            reproducible experimental scripts; outputs are ignored
models/                 compact served model artifacts and evaluation metadata
reports/                methodology notes and lightweight report inputs
archive/scripts/         retained legacy scripts for historical reference
share/                   self-contained dataset-building handoff
```

## Known limitations

- Fund membership and data-quality filters use each fund's available history, which introduces mild full-sample selection lookahead.
- Category metadata is current metadata applied across history; free point-in-time classifications are not available.
- Funds without a usable category cannot receive a peer-relative label.
- The Saudi section is a saved walk-forward evaluation record and must not be presented as a current recommendation.
- Model quality depends on public data availability, classification quality, and market regime stability.

## License and disclaimer

Released under the [MIT License](LICENSE).

This repository is for research and educational use only. It is not investment advice. Historical model performance and backtests do not guarantee future results.
