"""exp_14b_market_interact — market regime v2: per-fund INTERACTIONS only.

exp_14's ablation failure was an encoding problem, not a concept problem:
broadcast day-level states (same value for every fund that day) let trees
memorize time periods. This keeps only the cross-sectionally varying pieces
-- fund exposure x market state -- which carry the regime conditioning
without the date-fingerprint channel. (The probe itself had ranked
beta_x_eqvol as the top feature at 47% importance.) All strictly trailing.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, cat_pct_rank)

fields, cat = load_universe()
close = fields["Close"]
r1 = close.pct_change(fill_method=None)

mkt = r1.mean(axis=1)
spy = r1["SPY"] if "SPY" in r1.columns else mkt
tlt = r1["TLT"] if "TLT" in r1.columns else None

eq_trend = spy.rolling(20).sum()
eq_vol = spy.rolling(20).std()
rates_mom = tlt.rolling(20).sum() if tlt is not None else None

beta_60d = r1.rolling(60).cov(mkt).div(mkt.rolling(60).var(), axis=0)

feat_dict = {
    "beta_x_eqvol": beta_60d.mul(eq_vol, axis=0),
    "beta_x_eqtrend": beta_60d.mul(eq_trend, axis=0),
    "beta_x_eqvol_catrank": cat_pct_rank(beta_60d.mul(eq_vol, axis=0), cat),
}
if tlt is not None:
    dur_corr = r1.rolling(60).corr(tlt)
    feat_dict["dur_corr_60d"] = dur_corr
    feat_dict["durcorr_x_ratesmom"] = dur_corr.mul(rates_mom, axis=0)
    feat_dict["durcorr_catrank"] = cat_pct_rank(dur_corr, cat)

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_14b_market_interact", X)
run_experiment("exp_14b_market_interact", X, lbl, close.index)
