"""exp_12_temporal — trend-quality / autocorrelation / risk-adjusted momentum.

Existing momentum features are raw or peer-relative returns; none measure HOW a
fund got there (smooth trend vs noise), whether its daily returns mean-revert
or trend (autocorrelation), or momentum per unit of risk. All strictly
trailing (<= t).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, cat_pct_rank)

fields, cat = load_universe()
close = fields["Close"]

r1 = close.pct_change(fill_method=None)

# 1. lag-1 autocorrelation of daily returns over 60d (trend vs mean-revert)
autocorr1_60d = r1.rolling(60).corr(r1.shift(1))

# 2-3. risk-adjusted (sharpe-style) momentum, short and medium window
sharpe_mom_20d = close.pct_change(20, fill_method=None) / (r1.rolling(20).std() * np.sqrt(20))
sharpe_mom_60d = close.pct_change(60, fill_method=None) / (r1.rolling(60).std() * np.sqrt(60))

# 4. momentum acceleration: current 20d return vs the 20d return one month ago
ret_20d = close.pct_change(20, fill_method=None)
mom_accel_20d = ret_20d - ret_20d.shift(20)

# 5. fraction of up days in the last 20 (path smoothness)
updays_frac_20d = ((r1 > 0).astype(float).where(r1.notna())
                   .rolling(20, min_periods=15).mean())

# 6. signed trend strength: corr(log price, time)^2 * sign(slope) over 20d
logp = np.log(close)
tgrid = pd.DataFrame(np.tile(np.arange(len(close), dtype=float)[:, None],
                             (1, close.shape[1])),
                     index=close.index, columns=close.columns)
tr_corr = logp.rolling(20).corr(tgrid)
trend_str_20d = tr_corr.pow(2) * np.sign(tr_corr)

# 7. category rank of risk-adjusted momentum
sharpe_mom_catrank = cat_pct_rank(sharpe_mom_20d, cat)

feat_dict = {
    "autocorr1_60d": autocorr1_60d,
    "sharpe_mom_20d": sharpe_mom_20d,
    "sharpe_mom_60d": sharpe_mom_60d,
    "mom_accel_20d": mom_accel_20d,
    "updays_frac_20d": updays_frac_20d,
    "trend_str_20d": trend_str_20d,
    "sharpe_mom_catrank": sharpe_mom_catrank,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_12_temporal", X)
run_experiment("exp_12_temporal", X, lbl, close.index)
