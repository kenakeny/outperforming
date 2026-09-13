"""exp_06_catcontext — category-context features (regime/environment of the
fund's category; same value for every fund in a category on a given day,
except the interaction). Expected weak standalone vs a within-category label."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, loo_peer_mean, cat_pct_rank)

fields, cat = load_universe()
close = fields["Close"]

r1  = close.pct_change(fill_method=None)
r5  = close.pct_change(5, fill_method=None)
r20 = close.pct_change(20, fill_method=None)
v   = r1.rolling(20).std()

cat_disp_5d  = r5.T.groupby(cat).transform("std").T
cat_disp_20d = r20.T.groupby(cat).transform("std").T
cat_ret_5d   = r5.T.groupby(cat).transform("mean").T
cat_ret_20d  = r20.T.groupby(cat).transform("mean").T
cat_vol_20d  = v.T.groupby(cat).transform("mean").T
cat_n        = close.notna().T.groupby(cat).transform("sum").T
cat_breadth_5d = (r5 > 0).where(r5.notna()).T.groupby(cat).transform("mean").T
disp_x_vol_rank = cat_pct_rank(v, cat) * cat_disp_5d

feat_dict = {
    "cat_disp_5d":     cat_disp_5d,
    "cat_disp_20d":    cat_disp_20d,
    "cat_ret_5d":      cat_ret_5d,
    "cat_ret_20d":     cat_ret_20d,
    "cat_vol_20d":     cat_vol_20d,
    "cat_n":           cat_n,
    "cat_breadth_5d":  cat_breadth_5d,
    "disp_x_vol_rank": disp_x_vol_rank,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_06_catcontext", X)
run_experiment("exp_06_catcontext", X, lbl, close.index)
