MODEL_PATH = "model_bundle/stage1.ubj"   # .cbm / .ubj / .json / .joblib
YEAR       = 2026
HORIZON    = 20        # holding period in trading days
TOP_N      = 20       # funds long, and the same number short
MIN_ADV    = 50e6     # liquidity screen
COST_BPS   = 5.0      # one-way spread, charged on both legs

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt
from backtest_vs_spy import load_model, load_panel

model, feats = load_model(MODEL_PATH)
d = load_panel(feats, HORIZON, MIN_ADV)
d = d[d.index.get_level_values("date").year == YEAR].dropna(subset=feats, how="all")
p = model.predict_proba(d[feats])
sc = d.reset_index()[["date", "ticker", "fwd_ret"]]
sc["score"] = p[:, 2] - p[:, 0]

# rebalance every HORIZON days so holding periods don't overlap
days = np.array(sorted(sc["date"].unique()))[::HORIZON]
prev_l, prev_s, rows = set(), set(), []
for d0 in days:
    g = sc[sc["date"] == d0]
    if len(g) < 2 * TOP_N:
        continue
    longs, shorts = g.nlargest(TOP_N, "score"), g.nsmallest(TOP_N, "score")
    L, S = set(longs["ticker"]), set(shorts["ticker"])
    turn = (len(L ^ prev_l) + len(S ^ prev_s)) / (2 * TOP_N)
    prev_l, prev_s = L, S
    rows.append({"date": d0,
                 "long": longs["fwd_ret"].mean(), "short": shorts["fwd_ret"].mean(),
                 "gross": longs["fwd_ret"].mean() - shorts["fwd_ret"].mean(),
                 "bench": g["fwd_ret"].mean(), "cost": turn * COST_BPS / 1e4})
bt = pd.DataFrame(rows)
bt["net"] = bt["gross"] - bt["cost"]

# SPY over the identical windows
s = yf.download("SPY", start="2015-01-01", auto_adjust=True, progress=False)["Close"]
s = (s.iloc[:, 0] if hasattr(s, "columns") else s).dropna()
f = s.shift(-HORIZON) / s - 1.0
bt["spy"] = [f.iloc[min(s.index.searchsorted(pd.Timestamp(x)), len(s) - 1)] for x in bt["date"]]
bt = bt.dropna().reset_index(drop=True)

py = 252 / HORIZON
def stats(r, name):
    sd = r.std()
    dd = (np.cumsum(r) - np.maximum.accumulate(np.cumsum(r))).min()
    return {"series": name, "ann": r.mean() * py, "vol": sd * np.sqrt(py),
            "sharpe": r.mean() * py / (sd * np.sqrt(py)), "max_dd": dd,
            "hit": (r > 0).mean()}

tbl = pd.DataFrame([stats(bt["net"], f"long/short top{TOP_N}"),
                    stats(bt["long"], "long leg only"),
                    stats(-bt["short"], "short leg only"),
                    stats(bt["spy"], "SPY"),
                    stats(bt["bench"], "equal-weight universe")]).set_index("series")
beta, alpha = np.polyfit(bt["spy"], bt["net"], 1)
corr = np.corrcoef(bt["net"], bt["spy"])[0, 1]

print(f"{MODEL_PATH} | {YEAR} | {len(bt)} rebalances | {HORIZON}d holds | "
      f"{TOP_N} long / {TOP_N} short")
print(f"turnover {bt['cost'].mean()/COST_BPS*1e4:.2f}  cost {bt['cost'].mean()*1e4:.1f} bps/rebal\n")
print(tbl.assign(ann=tbl["ann"].map("{:+.2%}".format), vol=tbl["vol"].map("{:.2%}".format),
                 max_dd=tbl["max_dd"].map("{:.2%}".format), hit=tbl["hit"].map("{:.1%}".format),
                 sharpe=tbl["sharpe"].round(2)).to_string())
print(f"\nvs SPY: beta {beta:+.2f}  corr {corr:+.2f}  alpha {alpha*py:+.2%}/yr  "
      f"t={bt['net'].mean()/(bt['net'].std()/np.sqrt(len(bt))):+.2f}")
print("A long/short book is beta-neutral by construction -- judge it on Sharpe and\n"
      "correlation to SPY, not on whether it out-returns SPY.")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
dt = pd.to_datetime(bt["date"])
ax[0].plot(dt, np.cumsum(bt["net"]) * 100, lw=2, color="#2a78d6", label="long/short (net)")
ax[0].plot(dt, np.cumsum(bt["spy"]) * 100, lw=2, color="#111", label="SPY")
ax[0].plot(dt, np.cumsum(bt["bench"]) * 100, lw=1.5, ls="--", color="0.55", label="EW universe")
ax[0].axhline(0, color="#111", lw=1); ax[0].set_ylabel("cumulative %")
ax[0].set_title(f"Long/short top{TOP_N} vs SPY — {YEAR}"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(dt, np.cumsum(bt["long"]) * 100, lw=2, color="#1baf7a", label="long leg")
ax[1].plot(dt, np.cumsum(-bt["short"]) * 100, lw=2, color="#eb6834", label="short leg (inverted)")
ax[1].axhline(0, color="#111", lw=1); ax[1].set_title("Which leg carries the P&L?")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()