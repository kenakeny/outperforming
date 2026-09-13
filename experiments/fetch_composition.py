"""fetch_composition.py — pull ETF composition snapshots from yfinance.

One row per ticker in the modeling universe: 11 sector weights, asset-class
mix (stock/bond/cash/other), top-10 holdings concentration, and equity
valuation yields (P/E, P/B as reported). Saves incrementally to
data/raw/etf_composition.parquet every 100 tickers so it can be re-run to
resume after a crash / rate-limit.

NOTE: this is a TODAY-snapshot — yfinance has no historical composition.
Downstream these are static fund descriptors (like `category`), not
time-varying signals; they carry as-of-today bias for old sample dates.
"""
import sys, pathlib, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import yfinance as yf
from exp_harness import load_universe, RAW

OUT = RAW / "etf_composition.parquet"

SECTORS = ["realestate", "consumer_cyclical", "basic_materials",
           "consumer_defensive", "technology", "communication_services",
           "financial_services", "utilities", "industrials", "energy",
           "healthcare"]


def fetch_one(t):
    row = {"ticker": t}
    try:
        fd = yf.Ticker(t).funds_data
        sw = fd.sector_weightings or {}
        for s in SECTORS:
            row[f"sec_{s}"] = sw.get(s)
        ac = fd.asset_classes or {}
        row["pos_stock"] = ac.get("stockPosition")
        row["pos_bond"] = ac.get("bondPosition")
        row["pos_cash"] = ac.get("cashPosition")
        row["pos_other"] = (ac.get("otherPosition", 0) or 0) + \
                           (ac.get("preferredPosition", 0) or 0) + \
                           (ac.get("convertiblePosition", 0) or 0)
        th = fd.top_holdings
        if th is not None and len(th):
            row["top10_conc"] = float(th["Holding Percent"].sum())
            row["top1_weight"] = float(th["Holding Percent"].iloc[0])
        eq = fd.equity_holdings
        if eq is not None and len(eq.columns):
            col = eq.columns[0]
            row["earn_yield"] = pd.to_numeric(eq[col].get("Price/Earnings"), errors="coerce")
            row["book_yield"] = pd.to_numeric(eq[col].get("Price/Book"), errors="coerce")
        row["ok"] = 1
    except Exception as e:
        row["ok"] = 0
        row["err"] = str(e)[:80]
    return row


def main():
    fields, cat = load_universe()
    tickers = list(fields["Close"].columns)
    done = set()
    rows = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        done = set(prev["ticker"])
        print(f"resuming: {len(done)} already fetched")
    todo = [t for t in tickers if t not in done]
    print(f"fetching {len(todo)} of {len(tickers)} tickers")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 100 == 0:
                pd.DataFrame(rows).to_parquet(OUT, index=False)
                ok = sum(r.get("ok", 0) for r in rows)
                rate = i / (time.time() - t0)
                print(f"{i}/{len(todo)}  ok={ok}  {rate:.1f}/s  "
                      f"eta {((len(todo) - i) / rate / 60):.0f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    ok = df["ok"].sum() if "ok" in df else 0
    print(f"done: {len(df)} rows, {ok} ok -> {OUT}")


if __name__ == "__main__":
    main()
