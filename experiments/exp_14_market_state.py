"""exp_14_market_state — market regime features + fund-exposure interactions.

The 74 existing features are all fund-local; none say what the MARKET is
doing. Direction vs peers over 2 weeks is heavily driven by factor rotation
(the production model's own 2026 calls -- long duration, short leveraged
equity -- are one rotation bet). This family encodes the rotation state
explicitly, built ONLY from ETFs already inside the universe:

  market proxies: SPY (equities), TLT/SHY (rates/curve), HYG/LQD (credit),
  GLD (gold), UUP (dollar) -- with panel-derived fallbacks where missing.

Per-day states (broadcast to all funds) + per-fund interactions with
exposures (beta x equity trend, duration-corr x rates momentum). All
strictly trailing (<= t).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment)

fields, cat = load_universe()
close = fields["Close"]
r1 = close.pct_change(fill_method=None)


def proxy(ticker, fallback_frame=None):
    """Daily return series of a market-proxy ETF, or a panel fallback."""
    if ticker in r1.columns:
        return r1[ticker]
    print(f"proxy {ticker} missing -> fallback")
    return fallback_frame


mkt = r1.mean(axis=1)                       # equal-weight universe = market
spy = proxy("SPY", mkt)
tlt = proxy("TLT", None)
shy = proxy("SHY", None)
hyg = proxy("HYG", None)
lqd = proxy("LQD", None)
gld = proxy("GLD", None)
uup = proxy("UUP", None)

# ---------------------------------------------------------------- day states
def mom(s, n):
    return s.rolling(n).sum() if s is not None else None

states = {
    "mkt_eq_trend_20d": mom(spy, 20),
    "mkt_eq_vol_20d": spy.rolling(20).std(),
    "mkt_rates_mom_20d": mom(tlt, 20),
    "mkt_curve_mom_20d": (mom(tlt, 20) - mom(shy, 20)
                          if tlt is not None and shy is not None else None),
    "mkt_credit_mom_20d": (mom(hyg, 20) - mom(lqd, 20)
                           if hyg is not None and lqd is not None else None),
    "mkt_gold_mom_20d": mom(gld, 20) if gld is not None else None,
    "mkt_dollar_mom_20d": mom(uup, 20) if uup is not None else None,
    "mkt_breadth_50": (close > close.rolling(50).mean()).where(close.notna())
                      .mean(axis=1),
    "mkt_disp_5d": r1.std(axis=1).rolling(5).mean(),
}
states = {k: v for k, v in states.items() if v is not None}
print("day states:", list(states))

# ---------------------------------------------------------------- exposures
beta_60d = r1.rolling(60).cov(mkt).div(mkt.rolling(60).var(), axis=0)
dur_corr_60d = (r1.rolling(60).corr(tlt) if tlt is not None else None)

feat_dict = {}
for name, s in states.items():
    feat_dict[name] = pd.DataFrame(
        np.broadcast_to(s.to_numpy(dtype=float)[:, None], close.shape).copy(),
        index=close.index, columns=close.columns).where(close.notna())

feat_dict["beta_x_eqtrend"] = beta_60d.mul(states["mkt_eq_trend_20d"], axis=0)
feat_dict["beta_x_eqvol"] = beta_60d.mul(states["mkt_eq_vol_20d"], axis=0)
if dur_corr_60d is not None:
    feat_dict["dur_corr_60d"] = dur_corr_60d
    if "mkt_rates_mom_20d" in states:
        feat_dict["durcorr_x_ratesmom"] = dur_corr_60d.mul(
            states["mkt_rates_mom_20d"], axis=0)

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_14_market_state", X)
run_experiment("exp_14_market_state", X, lbl, close.index)
