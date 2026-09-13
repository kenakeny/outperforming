"""exp_04_liquidity — liquidity/size feature family.

Digs into liquidity properly: dollar-volume size (absolute + within-category),
Amihud illiquidity, volume trend/spike, dead-day fraction, turnover variability.
All features strictly trailing (data <= t only).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, loo_peer_mean, cat_pct_rank)

fields, cat = load_universe()
close, vol = fields["Close"], fields["Volume"]

dv = close * vol                      # dollar volume
r1 = close.pct_change(fill_method=None)

dv20 = dv.rolling(20).mean()

# 1-2. size: log dollar volume + category-relative rank
dollar_vol_log = np.log1p(dv20)
dollar_vol_catrank = cat_pct_rank(dv20, cat)

# 3-4. Amihud illiquidity (price impact per dollar traded) + category rank
amihud_20d = (r1.abs() / dv.replace(0, np.nan)).rolling(20).mean()
amihud_catrank = cat_pct_rank(amihud_20d, cat)

# 5-6. volume trend / spike vs 60d baseline
volume_trend = vol.rolling(20).mean() / vol.rolling(60).mean() - 1.0
volume_spike_5d = vol.rolling(5).mean() / vol.rolling(60).mean() - 1.0

# 7. fraction of dead/no-trade days (only where close is not NaN)
z = (vol == 0) & close.notna()
zero_vol_frac_20d = z.rolling(20).sum() / close.notna().rolling(20).sum()

# 8. dollar-volume variability
turnover_vol_20d = dv.rolling(20).std() / dv20

feat_dict = {
    "dollar_vol_log": dollar_vol_log,
    "dollar_vol_catrank": dollar_vol_catrank,
    "amihud_20d": amihud_20d,
    "amihud_catrank": amihud_catrank,
    "volume_trend": volume_trend,
    "volume_spike_5d": volume_spike_5d,
    "zero_vol_frac_20d": zero_vol_frac_20d,
    "turnover_vol_20d": turnover_vol_20d,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_04_liquidity", X)
run_experiment("exp_04_liquidity", X, lbl, close.index)
