"""
fetch_etf_news.py -- pull Polygon.io news for every ETF in the price panel.

For each ticker in data/raw/prices.parquet we request Polygon's
/v2/reference/news over that ETF's own history: [first_valid_price_date,
last_valid_price_date], with the end capped at MAX_END_DATE. Results are
written one-article-per-line to data/raw/etf_news.jsonl (tagged with the
query ticker), and a per-ticker log to data/raw/etf_news_status.csv.

The job is resumable: tickers already recorded in the status file are skipped
on restart, and a ticker's articles are only committed to the jsonl once it
has been fully paginated -- so a crash mid-ticker never leaves half-written
duplicates.

Auth: set POLYGON_API_KEY in the environment or a .env file in the repo root.

Usage:
    python fetch_etf_news.py                 # all tickers, resume-aware
    python fetch_etf_news.py --limit 10      # first 10 (smoke test)
    python fetch_etf_news.py --tickers SPY,QQQ,IWM
    python fetch_etf_news.py --rpm 100       # paid plan: 100 requests/min
    python fetch_etf_news.py --dry-run       # print plan, make no API calls

Note on coverage: Polygon's news history is much shorter than 2016 (typically
only the last few years, and thinner on a free plan). Requesting the full
range is harmless -- you just get back whatever exists.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import pandas as pd
import requests

NEWS_URL = "https://api.polygon.io/v2/reference/news"
MAX_END_DATE = "2026-07-01"          # hard cap on the query end date
PAGE_LIMIT = 1000                    # articles per page (Polygon max)


def resolve_root():
    root = pathlib.Path(__file__).resolve().parent
    return root


def load_api_key(root):
    key = os.environ.get("POLYGON_API_KEY")
    if key:
        return key.strip()
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("POLYGON_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(
        "POLYGON_API_KEY not found. Set it in your environment or a .env file:\n"
        "  PowerShell:  $env:POLYGON_API_KEY = 'your_key'\n"
        "  .env line:   POLYGON_API_KEY=your_key"
    )


def etf_date_ranges(root):
    """first/last valid price date per ticker, end capped at MAX_END_DATE."""
    prices = pd.read_parquet(root / "data" / "raw" / "prices.parquet").sort_index()
    cap = pd.Timestamp(MAX_END_DATE)
    rows = {}
    for t in prices.columns:
        s = prices[t]
        start, end = s.first_valid_index(), s.last_valid_index()
        if start is None or end is None:
            continue                      # no data at all -> nothing to query
        end = min(end, cap)
        if end < start:
            continue
        rows[t] = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return rows


class RateLimiter:
    """Guarantee a minimum spacing between requests (60 / rpm seconds)."""

    def __init__(self, rpm):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0

    def wait(self):
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


def fetch_ticker(ticker, start, end, api_key, limiter, session):
    """Return the full list of article dicts for one ticker (all pages)."""
    articles = []
    params = {
        "ticker": ticker,
        "published_utc.gte": start,
        "published_utc.lte": end,
        "order": "asc",
        "sort": "published_utc",
        "limit": PAGE_LIMIT,
        "apiKey": api_key,
    }
    url = NEWS_URL
    while True:
        limiter.wait()
        for attempt in range(6):
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:            # rate limited -> back off
                wait = min(60, 2 ** attempt * 5)
                print(f"    429 rate-limited, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f"{ticker}: gave up after repeated 429s")

        payload = resp.json()
        articles.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        if not next_url:
            break
        # next_url carries the cursor but not the apiKey; params must be reset.
        url, params = next_url, {"apiKey": api_key}
    return articles


def load_done(status_path):
    if not status_path.exists():
        return set()
    return set(pd.read_csv(status_path)["ticker"].astype(str))


def main():
    ap = argparse.ArgumentParser(description="Fetch Polygon news for all ETFs.")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tickers")
    ap.add_argument("--tickers", type=str, default=None, help="comma-separated subset")
    ap.add_argument("--rpm", type=int, default=5, help="requests per minute (free tier = 5)")
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

    news_path = root / "data" / "raw" / "etf_news.jsonl"
    status_path = root / "data" / "raw" / "etf_news_status.csv"

    done = load_done(status_path)
    todo = [t for t in tickers if t not in done]
    print(f"{len(tickers)} tickers selected | {len(done)} already done | {len(todo)} to fetch")

    if args.dry_run:
        for t in todo[:20]:
            print(f"  {t}: {ranges[t][0]} -> {ranges[t][1]}")
        if len(todo) > 20:
            print(f"  ... (+{len(todo) - 20} more)")
        eta_min = len(todo) * (60 / args.rpm) / 60
        print(f"dry-run only. rough ETA at {args.rpm} rpm: ~{eta_min:.1f} min (>= 1 request/ticker)")
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
            except Exception as e:                 # log and keep going
                print(f"[{i}/{len(todo)}] {t}: ERROR {e}", flush=True)
                status_f.write(f"{t},{start},{end},0,error\n")
                status_f.flush()
                continue

            for a in articles:                     # commit only after full fetch
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
