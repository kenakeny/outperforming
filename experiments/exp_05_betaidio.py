"""exp_05_betaidio — beta/residual features vs the leave-one-out category peer-mean.

The label is peer-relative, so we split each fund's return into a common
category component (loo peer-mean m) and an idiosyncratic remainder ex1 = r1 - m,
then build trailing beta / correlation / residual-momentum / idio-vol features.
All features use data <= t only.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, loo_peer_mean, cat_pct_rank)

fields, cat = load_universe()
close = fields["Close"]

r1 = close.pct_change(fill_method=None)
m = loo_peer_mean(r1, cat)          # fund-specific "category index" daily return
ex1 = r1 - m                        # peer-excess (idiosyncratic) daily return

# 1. rolling 60d beta of r1 vs m, per fund
beta_60d = r1.rolling(60).cov(m) / m.rolling(60).var().replace(0, np.nan)

# 2. category percentile rank of beta
beta_catrank = cat_pct_rank(beta_60d, cat)

# 3. rolling 60d correlation with the peer index
corr_cat_60d = r1.rolling(60).corr(m)

# 4/5. residual (peer-excess) return momentum
resid_ret_5d = ex1.rolling(5).sum()
resid_ret_20d = ex1.rolling(20).sum()

# 6. idiosyncratic vol
idio_vol_20d = ex1.rolling(20).std()

# 7. category percentile rank of idio vol
idio_vol_catrank = cat_pct_rank(idio_vol_20d, cat)

# 8. share of total vol that is idiosyncratic
idio_frac_20d = idio_vol_20d / r1.rolling(20).std().replace(0, np.nan)

# 9. t-stat of recent excess return
tstat_rel_5d = ex1.rolling(5).mean() / (ex1.rolling(5).std().replace(0, np.nan)
                                        / np.sqrt(5))

feat_dict = {
    "beta_60d": beta_60d,
    "beta_catrank": beta_catrank,
    "corr_cat_60d": corr_cat_60d,
    "resid_ret_5d": resid_ret_5d,
    "resid_ret_20d": resid_ret_20d,
    "idio_vol_20d": idio_vol_20d,
    "idio_vol_catrank": idio_vol_catrank,
    "idio_frac_20d": idio_frac_20d,
    "tstat_rel_5d": tstat_rel_5d,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_05_betaidio", X)
run_experiment("exp_05_betaidio", X, lbl, close.index)
