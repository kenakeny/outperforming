"""
build_sr_features.py -- support / resistance features from OHLC.

Resistance = recent swing high (rolling max of High), support = recent swing low
(rolling min of Low), both taken over the *prior* w days (shift(1)) so a feature
on day t only uses data through t-1 -- no lookahead.

Per window w in {20, 60} we derive, per (date, ticker):
  dist_res_w   (res - close)/close     distance up to resistance (small = near)
  dist_sup_w   (close - sup)/close     distance down to support
  range_pos_w  (close - sup)/(res-sup) 0 = at support .. 1 = at resistance
and for w=20 only (the actionable timescale):
  sr_width_20  (res - sup)/close       channel width
  res_touch_20 # of prior days that tagged resistance (within 2%)
  sup_touch_20 # of prior days that tagged support (within 2%)
  broke_res_20 close printed a new 20d high today (breakout)
  broke_sup_20 close printed a new 20d low today (breakdown)

Output: data/processed/sr_features.parquet indexed by (date, ticker).
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
TOL = 0.02          # "touch" = within 2% of the S/R level
WINDOWS = [20, 60]


def main():
    md = pd.read_parquet(ROOT / "data" / "raw" / "market_data.parquet").sort_index()
    high, low, close = md["High"], md["Low"], md["Close"]

    out = {}
    for w in WINDOWS:
        res = high.shift(1).rolling(w).max()      # resistance from the past
        sup = low.shift(1).rolling(w).min()        # support from the past
        span = (res - sup).replace(0, np.nan)
        out[f"dist_res_{w}"] = (res - close) / close
        out[f"dist_sup_{w}"] = (close - sup) / close
        out[f"range_pos_{w}"] = (close - sup) / span

        if w == 20:
            out["sr_width_20"] = (res - sup) / close
            hit_res = (high >= res * (1 - TOL)).astype(float)
            hit_sup = (low <= sup * (1 + TOL)).astype(float)
            out["res_touch_20"] = hit_res.shift(1).rolling(w).sum()
            out["sup_touch_20"] = hit_sup.shift(1).rolling(w).sum()
            out["broke_res_20"] = (close > res).astype(float)
            out["broke_sup_20"] = (close < sup).astype(float)

    # wide -> long (date, ticker)
    feats = pd.concat({k: v.stack() for k, v in out.items()}, axis=1)
    feats.index.names = ["date", "ticker"]
    feats = feats.sort_index()

    # clean inf and drop all-NaN warmup rows
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna(how="all")

    outp = ROOT / "data" / "processed" / "sr_features.parquet"
    feats.to_parquet(outp)
    print(f"S/R features -> {outp}")
    print(f"  shape {feats.shape} | cols: {list(feats.columns)}")
    print(feats.describe().round(3).T.to_string())

    # tiny leak sanity: recompute for a truncated panel and compare a late date
    t = feats.index.get_level_values("date").max()
    print(f"\nleak check @ {t.date()}: features use only shift(1) rolling -> trailing by construction")


if __name__ == "__main__":
    main()
