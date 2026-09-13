"""exp_03_peerext — peer-relative versions of the distance-from-own-extremes family.

Prior runs: px_to_52w_high / px_to_52w_low / px_to_sma_200 were strong in
ABSOLUTE form. This experiment adds category-percentile-rank and LOO-peer-mean
versions to test whether peer-relative encodings beat the absolute ones.
All features are strictly trailing (data <= t only).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, loo_peer_mean, cat_pct_rank)

fields, cat = load_universe()
close = fields["Close"]

roll_max_252 = close.rolling(252).max()
roll_min_252 = close.rolling(252).min()

px_to_52w_high = close / roll_max_252 - 1.0
px_to_52w_low  = close / roll_min_252 - 1.0
px_to_sma_200  = close / close.rolling(200).mean() - 1.0
px_to_sma_50   = close / close.rolling(50).mean() - 1.0
range_pos_52w  = (close - roll_min_252) / (roll_max_252 - roll_min_252)

feat_dict = {
    "px_to_52w_high":          px_to_52w_high,
    "px_to_52w_high_catrank":  cat_pct_rank(px_to_52w_high, cat),
    "px_to_52w_low":           px_to_52w_low,
    "px_to_52w_low_catrank":   cat_pct_rank(px_to_52w_low, cat),
    "px_to_sma_200":           px_to_sma_200,
    "px_to_sma_200_catrank":   cat_pct_rank(px_to_sma_200, cat),
    "px_to_sma_200_peerrel":   px_to_sma_200 - loo_peer_mean(px_to_sma_200, cat),
    "px_to_sma_50_catrank":    cat_pct_rank(px_to_sma_50, cat),
    "range_pos_52w":           range_pos_52w,
    "range_pos_52w_catrank":   cat_pct_rank(range_pos_52w, cat),
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_03_peerext", X)
run_experiment("exp_03_peerext", X, lbl, close.index)
