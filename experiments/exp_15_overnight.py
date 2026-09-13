"""exp_15_overnight — overnight vs intraday return decomposition.

We have Open prices and never used them for returns: the close-to-open gap
(overnight) and open-to-close (intraday) components of the same daily return
carry different information (institutional vs retail flow, news timing).
All strictly trailing (<= t).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, cat_pct_rank)

fields, cat = load_universe()
o, c = fields["Open"], fields["Close"]

gap = o / c.shift(1) - 1.0                       # overnight component
intra = c / o - 1.0                              # intraday component

on_ret_20d = gap.rolling(20).sum()
id_ret_20d = intra.rolling(20).sum()
denom = (gap.abs().rolling(20).sum() + intra.abs().rolling(20).sum())
on_share_20d = gap.abs().rolling(20).sum() / denom.replace(0, np.nan)
on_minus_id_20d = on_ret_20d - id_ret_20d
on_vol_20d = gap.rolling(20).std()
id_vol_20d = intra.rolling(20).std()

feat_dict = {
    "on_ret_20d": on_ret_20d,
    "id_ret_20d": id_ret_20d,
    "on_share_20d": on_share_20d,
    "on_minus_id_20d": on_minus_id_20d,
    "on_vol_ratio_20d": on_vol_20d / id_vol_20d.replace(0, np.nan),
    "on_ret_catrank": cat_pct_rank(on_ret_20d, cat),
    "id_ret_catrank": cat_pct_rank(id_ret_20d, cat),
}

X = assemble(feat_dict)
lbl = build_label(fields["Close"], cat)
save_features("exp_15_overnight", X)
run_experiment("exp_15_overnight", X, lbl, fields["Close"].index)
