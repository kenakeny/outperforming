"""exp_01_voltail — volatility tail-sharpening feature family.

Prior finding: vol is the dominant signal but only the top vol decile shows a
big positive peer-relative forward return; deciles 1-9 are flat. These
features try to isolate that tail explicitly. All strictly trailing (<= t).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, cat_pct_rank)

fields, cat = load_universe()
close, high, low = fields["Close"], fields["High"], fields["Low"]

r1 = close.pct_change(fill_method=None)

# 1. reference: 20d realized vol
vol_20d = r1.rolling(20).std()

# 2. rolling 252d percentile rank of vol_20d within the fund's own history
vol_pctile_1y = vol_20d.rolling(252).rank(pct=True)

# 3. coefficient of variation of vol over 60d
vol_of_vol_60d = vol_20d.rolling(60).std() / vol_20d.rolling(60).mean()

# 4. short vol vs long vol ratio
vol_spike = r1.rolling(5).std() / r1.rolling(60).std()

# 5. cross-sectional vol rank within category (0..1)
vol_cat_rank = cat_pct_rank(vol_20d, cat)

# 6. explicit tail flag: top decile of within-category vol rank
vol_top_decile = (vol_cat_rank >= 0.9).astype(float).where(vol_cat_rank.notna())

# 7. intraday range spike
hl = (high - low) / close
hl_spike = hl.rolling(5).mean() / hl.rolling(60).mean()

feat_dict = {
    "vol_20d": vol_20d,
    "vol_pctile_1y": vol_pctile_1y,
    "vol_of_vol_60d": vol_of_vol_60d,
    "vol_spike": vol_spike,
    "vol_cat_rank": vol_cat_rank,
    "vol_top_decile": vol_top_decile,
    "hl_spike": hl_spike,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_01_voltail", X)
run_experiment("exp_01_voltail", X, lbl, close.index)
