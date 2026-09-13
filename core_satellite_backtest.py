"""core_satellite_backtest.py -- "mostly conservative, some extreme" portfolio:

- CORE sleeve (85% of capital): funds the neutral-vs-extreme (15/70/15) model
  is most confident sit in the calm 70% neutral band -- low predicted P(extreme),
  equal-weighted.
- SATELLITE sleeve (15% of capital): funds that model flags as most likely
  "extreme" AND that the original 3-class model calls bullish (predicted
  "over"), equal-weighted.

Same 2026 test year, HORIZON=5, embargo, and feature set as everywhere else.
Compared against SPY over the identical rebalance windows.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from exp_harness import load_sweep_dataset

OUT = pathlib.Path("reports/model_eval")
HORIZON = 5
CORE_WEIGHT, SAT_WEIGHT = 0.85, 0.15
CORE_FRAC, SAT_FRAC = 0.40, 0.10     # fraction of daily universe in each sleeve
COST_BPS = 15.0

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
te = d.loc[te_mask].reset_index()

model3 = XGBClassifier(); model3.load_model("models/model_xgboost.ubj")
model_ne = XGBClassifier(); model_ne.load_model("models/model_xgboost_neutral_vs_extreme_15_70_15.ubj")

X_te = te[features]
proba3 = model3.predict_proba(X_te)
proba_ne = model_ne.predict_proba(X_te)[:, 1]     # P(extreme)

te["p_extreme"] = proba_ne
te["dir_score"] = proba3[:, 2] - proba3[:, 0]      # bullish - bearish
te["conservative_score"] = -proba_ne               # favor low P(extreme)
te["satellite_score"] = proba_ne * te["dir_score"].clip(lower=0)  # extreme AND bullish

days = np.array(sorted(te["date"].unique()))[::HORIZON]
prev, rows = {}, []
for d0 in days:
    g = te[te["date"] == d0]
    if len(g) < 10:
        continue
    n_core = max(int(round(len(g) * CORE_FRAC)), 5)
    n_sat = max(int(round(len(g) * SAT_FRAC)), 2)

    core = g.sort_values("conservative_score", ascending=False).head(n_core)
    sat_pool = g[~g["ticker"].isin(core["ticker"])]
    sat = sat_pool.sort_values("satellite_score", ascending=False).head(n_sat)

    w_core = pd.Series(CORE_WEIGHT / len(core), index=core["ticker"])
    w_sat = pd.Series(SAT_WEIGHT / len(sat), index=sat["ticker"]) if len(sat) else pd.Series(dtype=float)
    held = w_core.add(w_sat, fill_value=0.0).to_dict()

    turn = sum(abs(held.get(t, 0) - prev.get(t, 0)) for t in set(held) | set(prev))
    prev = held
    port = float(core["fwd_ret"].mul(w_core.values).sum() +
                 (sat["fwd_ret"].mul(w_sat.values).sum() if len(sat) else 0.0))
    bench = g["fwd_ret"].mean()
    cost = turn * COST_BPS / 1e4
    rows.append({"date": d0, "port": port, "bench": bench, "turn": turn, "cost": cost,
                "n_core": len(core), "n_sat": len(sat)})

bt = pd.DataFrame(rows)
py = 252 / HORIZON
net = bt["port"] - bt["cost"]


def spy_forward(dates, horizon, retries=3):
    import time
    import yfinance as yf
    for attempt in range(retries):
        s = yf.download("SPY", start="2015-01-01", auto_adjust=True, progress=False)
        if not s.empty:
            s = s["Close"]
            s = (s.iloc[:, 0] if hasattr(s, "columns") else s).dropna()
            if len(s) > horizon:
                f = s.shift(-horizon) / s - 1.0
                return np.array([f.iloc[min(s.index.searchsorted(pd.Timestamp(dt)),
                                            len(s) - 1)] for dt in dates], dtype=float)
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise SystemExit("could not download SPY from yfinance")


spy = spy_forward(bt["date"], HORIZON)


def ann_ir_t(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    a = r.mean() * py
    sd = r.std()
    ir = a / (sd * np.sqrt(py)) if sd else np.nan
    t = r.mean() / (sd / np.sqrt(len(r))) if sd else np.nan
    return a, ir, t


port_ann, port_ir, port_t = ann_ir_t(net)
spy_ann, spy_ir, spy_t = ann_ir_t(spy)
bench_ann, bench_ir, _ = ann_ir_t(bt["bench"])
ex = net.to_numpy() - spy
ex_ann, ex_ir, ex_t = ann_ir_t(ex)

print(f"rebalances: {len(bt)}  avg core n={bt['n_core'].mean():.0f}  avg sat n={bt['n_sat'].mean():.1f}  "
      f"avg turnover={bt['turn'].mean():.2f}")
print(f"\ncore-satellite (85/15): ann={port_ann:+.2%}  IR={port_ir:+.2f}  t={port_t:+.2f}")
print(f"SPY                   : ann={spy_ann:+.2%}  IR={spy_ir:+.2f}  t={spy_t:+.2f}")
print(f"universe (EW)         : ann={bench_ann:+.2%}  IR={bench_ir:+.2f}")
print(f"vs SPY excess         : {ex_ann:+.2%}/yr  IR={ex_ir:+.2f}  t={ex_t:+.2f}")

cum_port = (1 + net).cumprod()
cum_spy = (1 + pd.Series(spy).fillna(0)).cumprod()
cum_bench = (1 + bt["bench"]).cumprod()

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(bt["date"], cum_port, label="core-satellite (85% conservative / 15% extreme)", lw=2.2)
ax.plot(bt["date"], cum_spy, label="SPY", lw=1.8, ls="--")
ax.plot(bt["date"], cum_bench, label="universe (equal-weight)", lw=1.8, ls=":")
ax.set_title(f"Core-satellite (85/15) vs SPY -- 2026 test year\n"
            f"model ann={port_ann:+.2%} IR={port_ir:+.2f}   |   SPY ann={spy_ann:+.2%} IR={spy_ir:+.2f}",
            fontsize=10)
ax.set_ylabel("growth of $1"); ax.legend(fontsize=8.5); fig.autofmt_xdate()
fig.tight_layout()
out_path = OUT / "core_satellite_vs_spy.png"
fig.savefig(out_path, dpi=150)
print(f"\nsaved -> {out_path}")

pd.DataFrame([{"name": "core-satellite (85/15)", "ann": port_ann, "ir": port_ir, "t": port_t},
             {"name": "SPY", "ann": spy_ann, "ir": spy_ir, "t": spy_t},
             {"name": "universe (EW)", "ann": bench_ann, "ir": bench_ir, "t": np.nan}]
            ).to_csv(OUT / "core_satellite_vs_spy_summary.csv", index=False)
