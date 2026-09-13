"""Backtest figures. python reports/us_backtest/plot_backtest.py"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
HERE = pathlib.Path(__file__).resolve().parent
D = json.load(open(HERE / "chart_data.json"))

S, U = D["screened_50M"], D["unscreened"]
dates = pd.to_datetime(S["dates"])
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def cum(x):
    return np.cumsum(x)


fig, ax = plt.subplots(2, 3, figsize=(18, 9))

# 1. cumulative net excess
a = ax[0, 0]
a.plot(dates, cum(S["net"]) * 100, lw=2, color=BLUE, label="screened (>=$50M ADV, 396 funds)")
a.plot(dates, cum(U["net"]) * 100, lw=2, color=ORANGE, label="unscreened (all 1,578 funds)")
a.axhline(0, color="0.4", lw=1)
a.set_title("Cumulative NET excess return\n(same model — only tradeable universe differs)", fontsize=11)
a.set_ylabel("cumulative excess %")
a.legend(fontsize=8)

# 2. gross vs net, screened
a = ax[0, 1]
a.plot(dates, cum(S["gross"]) * 100, lw=2, color=BLUE, label="screened gross")
a.plot(dates, cum(S["net"]) * 100, lw=2, color=GREEN, label="screened net")
a.plot(dates, cum(U["gross"]) * 100, lw=1.5, color=ORANGE, ls="--", label="unscreened gross")
a.plot(dates, cum(U["net"]) * 100, lw=2, color=ORANGE, label="unscreened net")
a.axhline(0, color="0.4", lw=1)
a.set_title("Cost drag: gross vs net", fontsize=11)
a.set_ylabel("cumulative excess %")
a.legend(fontsize=8)

# 3. cost per rebalance
a = ax[0, 2]
a.plot(dates, U["cost"], lw=1.5, color=ORANGE, label="unscreened")
a.plot(dates, S["cost"], lw=1.5, color=BLUE, label="screened")
a.set_title("Cost per rebalance (bps)\nturnover nearly identical: 0.81 vs 0.76", fontsize=11)
a.set_ylabel("bps")
a.legend(fontsize=8)

# 4. rank IC by quarter
a = ax[1, 0]
q = pd.Series({pd.Timestamp(k): v for k, v in S["ic_monthly"].items()}).sort_index()
cols = [GREEN if v > 0 else ORANGE for v in q.values]
a.bar(q.index, q.values, width=70, color=cols)
a.axhline(0, color="0.4", lw=1)
a.axhline(q.mean(), color=BLUE, ls="--", lw=1.5, label=f"mean {q.mean():.4f}")
a.set_title("Rank IC by quarter (screened)", fontsize=11)
a.set_ylabel("per-day Spearman IC")
a.legend(fontsize=8)

# 5. per-year net excess
a = ax[1, 1]
df = pd.DataFrame({"date": dates, "s": S["net"], "u": U["net"]})
df["yr"] = df.date.dt.year
yr = df.groupby("yr")[["s", "u"]].sum() * 100
x = np.arange(len(yr))
a.bar(x - 0.2, yr["s"], 0.4, color=BLUE, label="screened")
a.bar(x + 0.2, yr["u"], 0.4, color=ORANGE, label="unscreened")
a.axhline(0, color="0.4", lw=1)
a.set_xticks(x)
a.set_xticklabels(yr.index, rotation=0)
a.set_title("Net excess return by year (%)", fontsize=11)
a.set_ylabel("excess %")
a.legend(fontsize=8)

# 6. breakeven vs assumed spread
a = ax[1, 2]
labels = ["screened", "unscreened"]
be = [S["stats"]["breakeven_spread_bps"], U["stats"]["breakeven_spread_bps"]]
asm = [S["stats"]["mean_spread_bps"], U["stats"]["mean_spread_bps"]]
x = np.arange(2)
a.bar(x - 0.2, be, 0.4, color=GREEN, label="breakeven spread")
a.bar(x + 0.2, asm, 0.4, color=ORANGE, label="assumed actual spread")
for i, (b, m) in enumerate(zip(be, asm)):
    a.text(i, max(b, m) + 2, f"{'PROFITABLE' if b > m else 'LOSES MONEY'}\n{b/m:.1f}x margin",
           ha="center", fontsize=9,
           color=("#0ca30c" if b > m else "#d03b3b"), fontweight="bold")
a.set_xticks(x)
a.set_xticklabels(labels)
a.set_ylim(0, max(be + asm) * 1.35)
a.set_title("Breakeven vs actual spread (bps)", fontsize=11)
a.set_ylabel("bps per side")
a.legend(fontsize=8)

for a in ax.flat:
    a.tick_params(labelsize=8)

plt.suptitle("US ETF signal — long-only, top-third, 20-day horizon, purged walk-forward 2019-2026",
             fontsize=13, y=1.00)
plt.tight_layout()
out = HERE / "backtest_overview.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)

# ---- second figure: equity curves in total-return terms ----
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
a = ax2[0]
port_s = cum(np.array(S["net"]) + np.array(S["bench"])) * 100
port_u = cum(np.array(U["net"]) + np.array(U["bench"])) * 100
bench = cum(S["bench"]) * 100
a.plot(dates, port_s, lw=2, color=BLUE, label="screened portfolio")
a.plot(dates, port_u, lw=2, color=ORANGE, label="unscreened portfolio")
a.plot(dates, bench, lw=2, color="0.45", ls="--", label="equal-weight benchmark")
a.set_title("Cumulative TOTAL return (net of costs)", fontsize=11)
a.set_ylabel("cumulative %")
a.legend(fontsize=9)

a = ax2[1]
s = pd.Series(np.array(S["net"]) * 100, index=dates)
dd = s.cumsum() - s.cumsum().cummax()
u = pd.Series(np.array(U["net"]) * 100, index=dates)
ddu = u.cumsum() - u.cumsum().cummax()
a.fill_between(dates, dd, 0, color=BLUE, alpha=0.5, label=f"screened (max {dd.min():.1f}%)")
a.fill_between(dates, ddu, 0, color=ORANGE, alpha=0.5, label=f"unscreened (max {ddu.min():.1f}%)")
a.set_title("Drawdown of the excess-return stream", fontsize=11)
a.set_ylabel("drawdown %")
a.legend(fontsize=9)
for a in ax2.flat:
    a.tick_params(labelsize=8)
plt.tight_layout()
out2 = HERE / "backtest_equity.png"
fig2.savefig(out2, dpi=120, bbox_inches="tight")
print("saved", out2)
