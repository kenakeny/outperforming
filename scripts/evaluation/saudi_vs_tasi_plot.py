"""Plot cumulative growth of the saudi_only portfolio vs
the TASI index (and the FALCOM Saudi Equity ETF) over the same test dates.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core
import saudi as saudi_mod

OUT = pathlib.Path("reports/model_eval")
ARM, HORIZON, COST_BPS = "saudi_only", 20, 15.0

p = pd.read_parquet("reports/saudi_predictions.parquet").loc[ARM]
bt = core.backtest(p, horizon=HORIZON, top=0.33, cost_bps=COST_BPS)

bench = saudi_mod.benchmarks(bt["date"], HORIZON)
cum_port = (1 + bt["port"] - bt["cost"]).cumprod()

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(bt["date"], cum_port, label=f"model ({ARM})", lw=2.2)
for name, r in bench.items():
    r = pd.Series(r, index=bt["date"])
    cum = (1 + r.fillna(0)).cumprod()
    ax.plot(bt["date"], cum, label=name, lw=1.8, ls="--")

py = 252 / HORIZON
st_model = core.stats(bt, HORIZON, col="port")
lines = [f"model  ann={st_model['ann']:+.2%}  IR={st_model['ir']:+.2f}"]
for name, r in bench.items():
    r = r[~np.isnan(r)]
    ann = r.mean() * py
    ir = ann / (r.std() * np.sqrt(py)) if r.std() else np.nan
    lines.append(f"{name}  ann={ann:+.2%}  IR={ir:+.2f}")

ax.set_title(f"Saudi transfer ({ARM}) vs TASI\n" + "\n".join(lines), fontsize=9)
ax.set_ylabel("growth of $1"); ax.legend(fontsize=8); fig.autofmt_xdate()
fig.tight_layout()
out_path = OUT / f"{ARM}_vs_tasi.png"
fig.savefig(out_path, dpi=150)
print(f"saved -> {out_path}")
print("\n".join(lines))
