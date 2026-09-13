"""exp_11_downside — drawdown / downside-risk / higher-moment feature family.

Prior findings: vol level and range features carry most of the signal, but none
of the existing 59 features describe the SHAPE of the return distribution
(asymmetry, tails) or the fund's drawdown state. Funds sitting deep in a
drawdown behave differently from funds at highs at the same vol level.
All strictly trailing (<= t).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, cat_pct_rank)

fields, cat = load_universe()
close, high, low = fields["Close"], fields["High"], fields["Low"]

r1 = close.pct_change(fill_method=None)

# 1-2. current drawdown from trailing 60d / 252d high
dd_60d  = close / close.rolling(60).max() - 1.0
dd_252d = close / close.rolling(252).max() - 1.0

# 3. days since the trailing 60d high (0 = at high today), NaN-safe via ffill
#    for the argmax only; masked back to NaN where close itself is missing.
W = 60
cf = close.ffill()
vals = cf.to_numpy()
days_since = np.full(vals.shape, np.nan)
if len(vals) >= W:
    win = np.lib.stride_tricks.sliding_window_view(vals, W, axis=0)  # (T-W+1, N, W)
    with np.errstate(invalid="ignore"):
        am = np.nanargmax(np.where(np.isnan(win), -np.inf, win), axis=-1)
    days_since[W - 1:] = (W - 1) - am
days_since_high_60d = pd.DataFrame(days_since, index=close.index,
                                   columns=close.columns).where(close.notna())

# 4. downside-vol share: semivol / total vol (asymmetry of recent risk)
semivol_20d = r1.where(r1 < 0).rolling(20, min_periods=5).std()
downside_share_20d = semivol_20d / r1.rolling(20).std()

# 5. sortino-style ratio: mean daily ret over downside vol
sortino_20d = r1.rolling(20).mean() / semivol_20d

# 6-7. higher moments of daily returns
skew_60d = r1.rolling(60).skew()
kurt_60d = r1.rolling(60).kurt()

# 8. drawdown rank within category (deep-drawdown funds vs their own peers)
dd_catrank = cat_pct_rank(dd_60d, cat)

feat_dict = {
    "dd_60d": dd_60d,
    "dd_252d": dd_252d,
    "days_since_high_60d": days_since_high_60d,
    "downside_share_20d": downside_share_20d,
    "sortino_20d": sortino_20d,
    "skew_60d": skew_60d,
    "kurt_60d": kurt_60d,
    "dd_catrank": dd_catrank,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_11_downside", X)
run_experiment("exp_11_downside", X, lbl, close.index)
