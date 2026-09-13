"""
score_sentiment.py -- FinBERT sentiment over the fetched ETF news, aggregated
into a per-(date, ticker) feature panel.

Pipeline:
  1. Read both news dumps (Polygon + Finnhub). Each article is deduped by id so
     FinBERT scores every unique article once (~65k), not once per tagged ETF
     (~349k rows).
  2. Score title+description / headline+summary with ProsusAI/finbert on GPU.
     Per-article signed score = P(positive) - P(negative)  in [-1, 1].
  3. Assign each article to the *next* trading day on/after (published + 1 day)
     -- so a feature on trading day T never sees news from T itself or later
     (no lookahead; consistent with the repo's leak-audit discipline).
  4. Aggregate per (trading_day, ticker) and add trailing-5d rollups.

Output: data/processed/news_features.parquet indexed by (date, ticker) with
  news_sent_1d, news_n_1d, news_sent_5d, news_n_5d
Raw per-article scores are cached to data/processed/news_scores_raw.parquet so
re-runs skip the (expensive) FinBERT pass.
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = "ProsusAI/finbert"
BATCH = 64
MAXLEN = 256

# source -> (path, id_key, (text_fields...), date_key)
SOURCES = {
    "polygon": ("data/raw/etf_news.jsonl", "id", ("title", "description"), "published_utc"),
    "finnhub": ("data/raw/etf_news_finnhub.jsonl", "id", ("headline", "summary"), "datetime"),
}


def parse_date(source, v):
    if v is None:
        return pd.NaT
    if source == "finnhub":
        return pd.to_datetime(int(v), unit="s").normalize()
    return pd.to_datetime(v, utc=True).tz_localize(None).normalize()


def load_rows():
    """Return (uniq_df[source,id,text], rows_df[source,id,ticker,pub_date])."""
    uniq, rows = {}, []
    for source, (path, idk, fields, datek) in SOURCES.items():
        p = ROOT / path
        if not p.exists():
            print(f"  skip {source}: {path} missing")
            continue
        n = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                aid = d.get(idk)
                if aid is None:
                    continue
                key = (source, str(aid))
                if key not in uniq:
                    text = " ".join(str(d.get(f, "") or "") for f in fields).strip()
                    uniq[key] = text
                rows.append((source, str(aid), d.get("query_ticker"),
                             parse_date(source, d.get(datek))))
                n += 1
        print(f"  {source}: {n} rows, {sum(1 for k in uniq if k[0]==source)} unique")
    uniq_df = pd.DataFrame(
        [(s, i, t) for (s, i), t in uniq.items()], columns=["source", "id", "text"]
    )
    rows_df = pd.DataFrame(rows, columns=["source", "id", "ticker", "pub_date"])
    return uniq_df, rows_df


def score_finbert(texts):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    # ProsusAI/finbert ships .bin weights; transformers>=5 blocks torch.load on
    # torch<2.6 (CVE-2025-32434), so load the safetensors revision instead.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, use_safetensors=True).to(dev).eval()
    # finbert label order: 0=positive, 1=negative, 2=neutral
    lbl = {v.lower(): k for k, v in model.config.id2label.items()}
    pos_i, neg_i = lbl["positive"], lbl["negative"]

    out = np.empty(len(texts), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(texts), BATCH):
        chunk = texts[s:s + BATCH]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True,
                  max_length=MAXLEN).to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        out[s:s + BATCH] = (probs[:, pos_i] - probs[:, neg_i]).cpu().numpy()
        if s % (BATCH * 50) == 0:
            done = s + len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"    scored {done}/{len(texts)}  ({rate:.0f}/s)", flush=True)
    return out


def main():
    print("loading news rows...")
    uniq_df, rows_df = load_rows()

    cache = ROOT / "data" / "processed" / "news_scores_raw.parquet"
    if cache.exists():
        print("using cached FinBERT scores")
        scored = pd.read_parquet(cache)
    else:
        print(f"scoring {len(uniq_df)} unique articles with FinBERT...")
        uniq_df["sent"] = score_finbert(uniq_df["text"].tolist())
        scored = uniq_df[["source", "id", "sent"]]
        scored.to_parquet(cache)
        print(f"cached raw scores -> {cache}")

    # attach scores to every (ticker, date) row
    rows = rows_df.merge(scored, on=["source", "id"], how="inner").dropna(
        subset=["ticker", "pub_date"])

    # assign each article to the next trading day strictly after publication
    prices = pd.read_parquet(ROOT / "data" / "raw" / "prices.parquet").sort_index()
    tidx = prices.index
    pos = tidx.searchsorted(rows["pub_date"].values + np.timedelta64(1, "D"), side="left")
    pos = np.clip(pos, 0, len(tidx) - 1)
    rows["date"] = tidx[pos]

    daily = (rows.groupby(["date", "ticker"])
             .agg(news_sent_1d=("sent", "mean"), news_n_1d=("sent", "size")))

    # trailing-5d rollups per ticker on the trading panel
    parts = []
    for tkr, g in daily.reset_index().groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        g["news_sent_5d"] = g["news_sent_1d"].rolling(5, min_periods=1).mean()
        g["news_n_5d"] = g["news_n_1d"].rolling(5, min_periods=1).sum()
        g["ticker"] = tkr
        parts.append(g.reset_index())
    feats = (pd.concat(parts, ignore_index=True)
             .set_index(["date", "ticker"]).sort_index())

    out = ROOT / "data" / "processed" / "news_features.parquet"
    feats.to_parquet(out)
    print(f"\nnews feature panel -> {out}")
    print(f"  shape {feats.shape} | tickers with news: {feats.index.get_level_values('ticker').nunique()}")
    print(feats.describe().round(3).to_string())


if __name__ == "__main__":
    main()
