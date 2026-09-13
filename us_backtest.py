"""
us_backtest.py -- does the US ETF signal survive execution?

    python us_backtest.py                          # default sweep
    python us_backtest.py --min-adv 50e6 --top 0.1

Spread is assigned from measured dollar volume. Corwin-Schultz was tried and
REJECTED -- it priced SPY at 24bps (true ~0.3bp) because it measures volatility,
not spread.
"""
import argparse

import numpy as np
import pandas as pd

import core

SPREADS = [(1e9, 1.0), (1e8, 3.0), (1e7, 10.0), (1e6, 30.0), (0.0, 80.0)]


def spread_bps(adv):
    return next(b for floor, b in SPREADS if adv >= floor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--min-adv", type=float, default=50e6)
    p.add_argument("--model", default="xgboost")
    p.add_argument("--scores", default="reports/us_backtest/scores_xgboost_h20.parquet",
                   help="reuse cached scores; pass '' to re-run the walk-forward")
    a = p.parse_args()

    if a.scores:
        sc = pd.read_parquet(a.scores)
    else:
        d, tidx = core.panel(a.horizon)
        cols = [c for c in d.columns if c not in core.LABEL_COLS]
        sc = core.walkforward(d, cols, tidx, a.horizon, a.model)
        sc.to_parquet(f"reports/us_backtest/scores_{a.model}_h{a.horizon}.parquet")

    sc = sc[sc["adv"] >= a.min_adv]
    bps = sc["adv"].map(spread_bps).mean()
    print(f"universe {sc.ticker.nunique()} funds | assumed spread {bps:.1f} bps\n")
    print(f"{'weight':8} {'top':>5} {'ann':>8} {'IR':>7} {'t':>7} {'turn':>6} {'breakeven':>10}")
    for weight in ("equal", "score"):
        for top in (0.10, 0.20, 0.33):
            bt = core.backtest(sc, a.horizon, top, cost_bps=bps, weight=weight)
            s = core.stats(bt, a.horizon)
            gross = (bt["port"] - bt["bench"]).mean()
            be = gross / max(bt["turn"].mean(), 1e-9) * 1e4
            print(f"{weight:8} {top:5.0%} {s['ann']:+8.2%} {s['ir']:+7.2f} "
                  f"{s['t']:+7.2f} {s['turn']:6.2f} {be:9.1f}b")


if __name__ == "__main__":
    main()
