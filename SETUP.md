# Running this

Two pieces: a Python/FastAPI backend (models + data already included below) and a
React/Vite frontend. Needs Python 3.10+ and Node 18+.

## 1. Backend

```
pip install -r requirements.txt
uvicorn serve.api:app --port 8000
```

Check it's alive: http://localhost:8000/health should say `"status": "ok"`.
Only 6 of the models in `serve/inference.py`'s registry are included
(catboost_sr[_news], xgboost_sr[_news], logreg_sr[_news]) -- that's what the
US screener needs. The Saudi tab needs nothing else; its data
(`reports/saudi_predictions.parquet`) is a pre-computed walk-forward record,
not a live model -- see its own in-app disclaimer.

## 2. Frontend

```
cd web
npm install
npm run build
```

That writes `web/dist`, which `serve/api.py` mounts at `/` -- so once it's
built, reloading http://localhost:8000 in a browser serves the whole app
(both the US Markets and Saudi Market tabs), no separate frontend server
needed.

(For active frontend development instead of a one-time build: run the
backend on 8000 as above, then `npm run dev` in `web/` for hot reload on
http://localhost:5173 -- vite proxies API calls to 8000.)

## 3. Tests (optional)

```
pytest
```

## What's NOT included (and why)

- `.env` -- may hold API keys, deliberately excluded, ask the person who sent
  this if you need it (only needed for `fetch_etf_news*.py`, not for serving).
- `web/node_modules` -- reinstalled by `npm install` above, not portable
  across machines/OSes anyway.
- Notebooks, `experiments/`, `archive/`, unregistered model variants,
  raw news/OHLCV pulls, and anything not on the path `serve/api.py` actually
  reads -- this is a runnable slice of a much bigger research repo, not the
  whole thing.
