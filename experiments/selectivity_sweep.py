"""selectivity_sweep — how much do the ratios improve if we get pickier?

Prior result (cost_sensitive_analysis.py): gating the ensemble's directional
calls to the top 15%/day by conviction gave 49% win rate and a thin payoff
edge -- not good enough. The lever not yet pulled: tighten the gate. Fewer,
higher-conviction calls should trade quantity for quality.

Sweeps top-K% from 30% down to 0.5% (same 1y-window ensemble, same 2026
test, same conviction ranking as the gated cost analysis: how much better
than the neutral default a directional call is under the aggressive cost
matrix). For each K reports:

  n_calls           total directional calls across the test year
  win_rate          fraction where predicted direction matched true label
  mean_payoff       mean signed 5d peer-relative excess (the money metric)
  win_loss_ratio    mean(payoff | payoff>0) / mean(|payoff| | payoff<0)
  sharpe            mean_payoff / std(payoff) -- risk-adjusted, not annualized
  breakeven_wr      win rate needed to break even given the realized
                    win/loss ratio -- compare directly to win_rate
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, EMBARGO, gpu_lock, EXP

TEST_YEAR, TOP_PCT = 2026, 0.15
KS = [0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]

COST_AGGRESSIVE = np.array([[0, 2, 6],
                            [1, 0, 1],
                            [6, 2, 0]], dtype=float)

d, features, _, _, _, trading_index = load_sweep_dataset()
date_index = d.index.get_level_values("date")

day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y_top"] = (day_pct >= 1.0 - TOP_PCT).astype("int8")
d["y_ext"] = ((day_pct >= 1.0 - TOP_PCT) | (day_pct <= TOP_PCT)).astype("int8")


def window_masks(years):
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
    test_end = pd.Timestamp(f"{TEST_YEAR}-12-31")
    cut = trading_index[max(trading_index.searchsorted(test_start) - EMBARGO, 0)]
    start = test_start - pd.DateOffset(years=years)
    tr = (date_index <= cut) & (date_index >= start)
    te = (date_index >= test_start) & (date_index <= test_end)
    tr_dates = np.sort(date_index[tr].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.90)]
    val_cut = trading_index[max(trading_index.searchsorted(val_start) - EMBARGO, 0)]
    fit = tr & (date_index <= val_cut)
    val = tr & (date_index >= val_start)
    return fit, val, te

fit_m, val_m, te_m = window_masks(1)
print(f"1y window: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}", flush=True)

from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, learning_rate=0.03, max_depth=8,
            min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, tree_method="hist", device="cuda",
            early_stopping_rounds=150, random_state=42, verbosity=0)

ft, vt = fit_m & (d["y_ext"] == 1).values, val_m & (d["y_ext"] == 1).values
with gpu_lock():
    stage2 = XGBClassifier(**BASE, objective="binary:logistic", eval_metric="auc")
    stage2.fit(d.loc[ft, features], d.loc[ft, "y_top"],
               eval_set=[(d.loc[vt, features], d.loc[vt, "y_top"])], verbose=False)

y_ext_fit = d.loc[fit_m, "y_ext"]
spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
with gpu_lock():
    stage1 = XGBClassifier(**BASE, objective="binary:logistic",
                           scale_pos_weight=spw, eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], y_ext_fit,
               eval_set=[(d.loc[val_m, features], d.loc[val_m, "y_ext"])], verbose=False)

p_ext = stage1.predict_proba(d.loc[te_m, features])[:, 1]
p_dir = stage2.predict_proba(d.loc[te_m, features])[:, 1]
P_ens = np.column_stack([p_ext * (1 - p_dir), 1 - p_ext, p_ext * p_dir])

cost_ud = np.minimum(P_ens @ COST_AGGRESSIVE[:, 0], P_ens @ COST_AGGRESSIVE[:, 2])
cost_neutral = P_ens @ COST_AGGRESSIVE[:, 1]
saving = cost_neutral - cost_ud
direction = np.where((P_ens @ COST_AGGRESSIVE[:, 2]) < (P_ens @ COST_AGGRESSIVE[:, 0]), 2, 0)

te = pd.DataFrame({"saving": saving, "direction": direction,
                   "y_true": d.loc[te_m, "target"].to_numpy(),
                   "rel": d.loc[te_m, "rel"].to_numpy()},
                  index=d.index[te_m])

rows = []
for k in KS:
    picks = (te.groupby(level="date")
             .apply(lambda g: g.nlargest(max(int(len(g) * k), 1), "saving")))
    picks = picks.droplevel(0) if picks.index.nlevels > 2 else picks
    win = (picks["direction"] == picks["y_true"]).to_numpy()
    sign = np.where(picks["direction"] == 2, 1, -1)
    payoff = sign * picks["rel"].to_numpy()

    wins, losses = payoff[payoff > 0], payoff[payoff < 0]
    wl_ratio = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan
    win_rate = float(win.mean())
    breakeven_wr = 1.0 / (1.0 + wl_ratio) if wl_ratio and not np.isnan(wl_ratio) else np.nan
    sharpe = float(payoff.mean() / payoff.std()) if payoff.std() > 0 else np.nan

    rows.append({"top_k_pct": k, "n_calls": len(picks),
                "win_rate": round(win_rate, 4),
                "mean_payoff": round(float(payoff.mean()), 5),
                "win_loss_ratio": round(wl_ratio, 3) if wl_ratio else None,
                "breakeven_wr": round(breakeven_wr, 4) if breakeven_wr else None,
                "sharpe": round(sharpe, 4)})
    print(f"K={k:>6.1%}  n={len(picks):>6}  win_rate={win_rate:.4f}  "
         f"payoff={payoff.mean():.5f}  wl_ratio={wl_ratio:.3f}  "
         f"breakeven_wr={breakeven_wr:.4f}  sharpe={sharpe:.4f}", flush=True)

res = pd.DataFrame(rows)
print("\n===== selectivity sweep (ensemble, 1y window, 2026 test) =====")
print(res.to_string(index=False))
res.to_csv(EXP / "results" / "selectivity_sweep.csv", index=False)
print("\nsaved -> experiments/results/selectivity_sweep.csv")
