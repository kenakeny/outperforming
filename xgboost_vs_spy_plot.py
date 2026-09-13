"""xgboost_vs_spy_plot.py -- cumulative growth of the full-feature XGBoost US
model's 2026 test-year portfolio vs buying SPY over the identical windows.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import core
from exp_harness import load_sweep_dataset

OUT = pathlib.Path("reports/model_eval")
HORIZON, TOP, COST_BPS = 5, 0.33, 15.0

d, features, fit_mask, val_mask, te_mask, _ = load_sweep_dataset()
model = XGBClassifier()
model.load_model("models/model_xgboost.ubj")

te = d.loc[te_mask]
proba = model.predict_proba(te[features])
scores = (te.reset_index()[["date", "ticker", "fwd_ret", "adv"]]
          if "adv" in te.columns else te.reset_index()[["date", "ticker", "fwd_ret"]].assign(adv=1.0))
scores["score"] = proba[:, 2] - proba[:, 0]

bt = core.backtest(scores, horizon=HORIZON, top=TOP, cost_bps=COST_BPS)


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
py = 252 / HORIZON
cum_port = (1 + bt["port"] - bt["cost"]).cumprod()
cum_bench = (1 + bt["bench"]).cumprod()
cum_spy = (1 + pd.Series(spy).fillna(0)).cumprod()

port = (bt["port"] - bt["cost"]).to_numpy()
ex = port - spy
st_model = core.stats(bt, HORIZON, col="port")


def ann_ir(r):
    r = r[~np.isnan(r)]
    a = r.mean() * py
    i = a / (r.std() * np.sqrt(py)) if r.std() else np.nan
    return a, i


spy_ann, spy_ir = ann_ir(spy)
bench_ann, bench_ir = ann_ir(bt["bench"].to_numpy())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(bt["date"], cum_port, label="XGBoost model (full features)", lw=2.2)
ax.plot(bt["date"], cum_spy, label="SPY", lw=1.8, ls="--")
ax.plot(bt["date"], cum_bench, label="universe (equal-weight)", lw=1.8, ls=":")

title = (f"XGBoost (full features) vs SPY -- 2026 test year\n"
         f"model  ann={st_model['ann']:+.2%}  IR={st_model['ir']:+.2f}   |   "
         f"SPY  ann={spy_ann:+.2%}  IR={spy_ir:+.2f}")
ax.set_title(title, fontsize=10)
ax.set_ylabel("growth of $1"); ax.legend(fontsize=9); fig.autofmt_xdate()
fig.tight_layout()
out_path = OUT / "xgboost_us_full_vs_spy.png"
fig.savefig(out_path, dpi=150)

print(f"saved -> {out_path}")
print(f"model : ann={st_model['ann']:+.2%}  IR={st_model['ir']:+.2f}")
print(f"SPY   : ann={spy_ann:+.2%}  IR={spy_ir:+.2f}")
print(f"universe (EW): ann={bench_ann:+.2%}  IR={bench_ir:+.2f}")
print(f"model vs SPY excess: {ex.mean()*py:+.2%}/yr  "
      f"IR {ex.mean()*py/(ex.std()*np.sqrt(py)):+.2f}  "
      f"t={ex.mean()/(ex.std()/np.sqrt(len(ex))):+.2f}")
