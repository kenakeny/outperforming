"""exp_07_range — range-based volatility estimators (Parkinson / Garman-Klass)
plus intraday-position (CLV) and overnight-gap features.

Motivation: in a raw-OHLCV model, High and Low carried ~75% of importance —
the tree was reconstructing intraday range as a volatility proxy. This family
extracts that signal cleanly with proper range-based estimators, which are
more statistically efficient than close-to-close std.

All features strictly trailing (data <= t only).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, loo_peer_mean, cat_pct_rank)

fields, cat = load_universe()
openp, high, low, close = fields["Open"], fields["High"], fields["Low"], fields["Close"]

# --- guarded building blocks -------------------------------------------------
low_safe   = low.replace(0, np.nan)
openp_safe = openp.replace(0, np.nan)
close_safe = close.replace(0, np.nan)

# log ratios, arguments clipped to positive
log_hl = np.log((high / low_safe).clip(lower=1e-12))       # log(H/L)
log_co = np.log((close / openp_safe).clip(lower=1e-12))    # log(C/O)

# 1. Parkinson volatility (20d)
pk = (log_hl ** 2) / (4 * np.log(2))
parkinson_20d = np.sqrt(pk.rolling(20).mean())

# 2. Garman-Klass volatility (20d)
gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
gk_20d = np.sqrt(gk.clip(lower=0).rolling(20).mean())

# 3. Parkinson vol, percentile-ranked within category (cross-sectional)
parkinson_catrank = cat_pct_rank(parkinson_20d, cat)

# 4. Close location value (20d mean) — where the close lands in the day's range
clv = (2 * close - high - low) / (high - low).replace(0, np.nan)
clv_20d = clv.rolling(20).mean()

# 5/6. Overnight gap: mean and mean-absolute (20d)
gap = openp / close.shift(1).replace(0, np.nan) - 1.0
gap_20d = gap.rolling(20).mean()
gap_abs_20d = gap.abs().rolling(20).mean()

# 7. Range expansion/contraction: 5d vs 60d mean relative range
rel_range = (high - low) / close_safe
range_ratio_5_60 = rel_range.rolling(5).mean() / rel_range.rolling(60).mean().replace(0, np.nan)

# 8. Range vol vs close-to-close vol — gappy/jumpy trading detector
ccvol = close.pct_change(fill_method=None).rolling(20).std()
pk_to_ccvol = parkinson_20d / ccvol.replace(0, np.nan)

feat_dict = {
    "parkinson_20d":     parkinson_20d,
    "gk_20d":            gk_20d,
    "parkinson_catrank": parkinson_catrank,
    "clv_20d":           clv_20d,
    "gap_20d":           gap_20d,
    "gap_abs_20d":       gap_abs_20d,
    "range_ratio_5_60":  range_ratio_5_60,
    "pk_to_ccvol":       pk_to_ccvol,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_07_range", X)
run_experiment("exp_07_range", X, lbl, close.index)
