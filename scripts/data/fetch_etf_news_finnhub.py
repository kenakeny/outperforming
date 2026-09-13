"""
Pull Finnhub news for every ETF in the price panel.

Finnhub counterpart to fetch_etf_news.py (Polygon). Same job, same outputs shape,
same resume behaviour -- only the API differs:

  * endpoint  /api/v1/company-news?symbol=X&from=YYYY-MM-DD&to=YYYY-MM-DD
  * auth      ?token=...  (env FINNHUB_API_KEY or a .env line)
  * no pagination, but each request covers at most ~1 year -> we split each ETF's
    [start, end] into <=365-day windows and dedupe articles by their Finnhub id.
  * free tier is 60 req/min (vs Polygon's 5), so --rpm defaults to 60.

The per-ETF date ranges, rate limiter, and resume/status handling are imported
from fetch_etf_news.py so the two scripts can't drift apart.

Outputs (kept separate from the Polygon run):
  data/raw/etf_news_finnhub.jsonl         one article/line, tagged query_ticker
  data/raw/etf_news_finnhub_status.csv    per-ticker log (resume source)

Usage:
    python -m scripts.data.fetch_etf_news_finnhub
    python -m scripts.data.fetch_etf_news_finnhub --limit 10
    python -m scripts.data.fetch_etf_news_finnhub --tickers SPY,QQQ,IWM
    python -m scripts.data.fetch_etf_news_finnhub --rpm 30
    python -m scripts.data.fetch_etf_news_finnhub --dry-run

Note: Finnhub's free company-news only serves roughly the last ~1 year of
history, so old start dates return whatever exists in that window -- expect
plenty of zero/low-count ETFs, same as Polygon.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd
import requests

from scripts.data.fetch_etf_news import etf_date_ranges, load_done, RateLimiter, resolve_root

NEWS_URL = "https://finnhub.io/api/v1/company-news"
WINDOW_DAYS = 365                    # Finnhub caps each request at ~1 year


def load_api_key(root):
    key = os.environ.get("FINNHUB_API_KEY")
    if key:
        return key.strip()
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("FINNHUB_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(
        "FINNHUB_API_KEY not found. Set it in your environment or a .env file:\n"
        "  PowerShell:  $env:FINNHUB_API_KEY = 'your_key'\n"
        "  .env line:   FINNHUB_API_KEY=your_key"
    )


def date_windows(start, end):
    """Split [start, end] (YYYY-MM-DD strings) into <=WINDOW_DAYS chunks."""
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    step = dt.timedelta(days=WINDOW_DAYS - 1)
    out = []
    cur = s
    while cur <= e:
        chunk_end = min(cur + step, e)
        out.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + dt.timedelta(days=1)
    return out


def fetch_ticker(ticker, start, end, api_key, limiter, session):
    """All articles for one ticker across yearly windows, deduped by id."""
    by_id = {}
    for w_start, w_end in date_windows(start, end):
        params = {"symbol": ticker, "from": w_start, "to": w_end, "token": api_key}
        limiter.wait()
        for attempt in range(6):
            resp = session.get(NEWS_URL, params=params, timeout=30)
            if resp.status_code == 429:              # rate limited -> back off
                wait = min(60, 2 ** attempt * 5)
                print(f"    429 rate-limited, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f"{ticker}: gave up after repeated 429s")

        for a in resp.json():                        # response is a plain list
            # 'id' is Finnhub's article id; fall back to url if ever missing.
            by_id[a.get("id", a.get("url"))] = a
    return list(by_id.values())


def main():
    ap = argparse.ArgumentParser(description="Fetch Finnhub news for all ETFs.")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tickers")
    ap.add_argument("--tickers", type=str, default=None, help="comma-separated subset")
    ap.add_argument("--rpm", type=int, default=60, help="requests per minute (free tier = 60)")
    ap.add_argument("--dry-run", action="store_true", help="print plan, no API calls")
    args = ap.parse_args()

    root = resolve_root()
    ranges = etf_date_ranges(root)

    if args.tickers:
        want = [t.strip().upper() for t in args.tickers.split(",")]
        ranges = {t: ranges[t] for t in want if t in ranges}
    tickers = sorted(ranges)
    if args.limit:
        tickers = tickers[: args.limit]

    news_path = root / "data" / "raw" / "etf_news_finnhub.jsonl"
    status_path = root / "data" / "raw" / "etf_news_finnhub_status.csv"

    done = load_done(status_path)
    todo = [t for t in tickers if t not in done]
    print(f"{len(tickers)} tickers selected | {len(done)} already done | {len(todo)} to fetch")

    if args.dry_run:
        for t in todo[:20]:
            wins = date_windows(*ranges[t])
            print(f"  {t}: {ranges[t][0]} -> {ranges[t][1]}  ({len(wins)} window(s))")
        if len(todo) > 20:
            print(f"  ... (+{len(todo) - 20} more)")
        total_reqs = sum(len(date_windows(*ranges[t])) for t in todo)
        eta_min = total_reqs * (60 / args.rpm) / 60
        print(f"dry-run only. ~{total_reqs} requests, ETA at {args.rpm} rpm: ~{eta_min:.1f} min")
        return

    api_key = load_api_key(root)
    limiter = RateLimiter(args.rpm)
    session = requests.Session()

    news_f = news_path.open("a", encoding="utf-8")
    new_status = not status_path.exists()
    status_f = status_path.open("a", encoding="utf-8")
    if new_status:
        status_f.write("ticker,start,end,n_articles,status\n")

    try:
        for i, t in enumerate(todo, 1):
            start, end = ranges[t]
            try:
                articles = fetch_ticker(t, start, end, api_key, limiter, session)
            except Exception as e:                   # log and keep going
                print(f"[{i}/{len(todo)}] {t}: ERROR {e}", flush=True)
                status_f.write(f"{t},{start},{end},0,error\n")
                status_f.flush()
                continue

            for a in articles:                       # commit only after full fetch
                a["query_ticker"] = t
                news_f.write(json.dumps(a, ensure_ascii=False) + "\n")
            news_f.flush()
            status_f.write(f"{t},{start},{end},{len(articles)},ok\n")
            status_f.flush()
            print(f"[{i}/{len(todo)}] {t}: {len(articles)} articles", flush=True)
    finally:
        news_f.close()
        status_f.close()

    print(f"\ndone. articles -> {news_path}")
    print(f"      status   -> {status_path}")


if __name__ == "__main__":
    main()
