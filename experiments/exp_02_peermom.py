"""exp_02_peermom — peer-relative momentum family.

Tests whether momentum measured RELATIVE TO CATEGORY PEERS carries signal
for the peer-relative tercile label (absolute momentum scored near-zero
importance in prior runs). 9 features, all strictly trailing (data <= t).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, loo_peer_mean, cat_pct_rank)

fields, cat = load_universe()
close = fields["Close"]

feat_dict = {}

# 1 & 2: peer-relative excess return and within-category percentile rank
for n in (5, 20, 60):
    ret_n = close.pct_change(n, fill_method=None)
    feat_dict[f"ret_{n}d_peerrel"] = ret_n - loo_peer_mean(ret_n, cat)
    feat_dict[f"ret_{n}d_catrank"] = cat_pct_rank(ret_n, cat)

# 3: skip-week momentum (t-25 -> t-5), peer-relative
r = close.shift(5) / close.shift(25) - 1.0
feat_dict["ret_20d_skip5_peerrel"] = r - loo_peer_mean(r, cat)

# 4: fraction of up days over 20d, ranked vs peers
up_frac = (close.pct_change(fill_method=None) > 0).rolling(20).mean()
feat_dict["mom_consistency_20d"] = cat_pct_rank(up_frac, cat)

# 5: rolling 5-day sum of daily excess-vs-peer returns
d1 = close.pct_change(fill_method=None)
ex1 = d1 - loo_peer_mean(d1, cat)
feat_dict["streak_rel_5d"] = ex1.rolling(5).sum()

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_02_peermom", X)
run_experiment("exp_02_peermom", X, lbl, close.index)
