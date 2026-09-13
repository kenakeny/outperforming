"""combined_best — 2-week horizon x tuned direction hyperparameters.

Two independent wins so far, never combined:
  - hp_tune_direction.py: depth=6, lr=0.03, min_child_weight=8, subsample=0.8,
    colsample=0.8, l2=5 beat the old defaults on the 5-day direction model
    (gated win_rate 0.548 vs 0.518, payoff +1.34% vs +0.91% at K=3%).
  - horizon_2week.py: switching the label to a 10-day (2-week) forward
    horizon beat 5-day even with default hyperparameters (payoff +1.54% vs
    +0.91% at K=3%).

This applies the tuned hyperparameters to the 2-week-horizon direction model
to see if the two effects stack. 1y training window, 2026 test only.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from exp_harness import load_universe, gpu_lock, EXP, SWEEP_FAMILIES

HORIZON, SKIP = 10, 0
EMBARGO = HORIZON + SKIP
TEST_YEAR, TOP_PCT = 2026, 0.15
KS = [0.10, 0.05, 0.03, 0.02, 0.01]
MC_NAMES = ["under", "neutral", "over"]

TUNED = dict(max_depth=6, learning_rate=0.03, min_child_weight=8,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)

def stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s

def loo_peer_mean(x, cat):
    s = x.T.groupby(cat).transform("sum").T
    n = x.notna().T.groupby(cat).transform("sum").T
    return (s - x.fillna(0)) / (n - x.notna().astype(int)).replace(0, np.nan)

def build_label_h(close, cat, horizon, skip):
    fwd = close.shift(-horizon) / close.shift(-skip) - 1.0
    rel = fwd - loo_peer_mean(fwd, cat)
    fwd_l, rel_l = stack(fwd), stack(rel)
    target = (rel_l.dropna().groupby(level="date")
             .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                          labels=[0, 1, 2])).astype("int8"))
    return (pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1)
           .dropna(subset=["target"]))

frames = [pd.read_parquet(EXP / "features" / f) for f in SWEEP_FAMILIES
         if (EXP / "features" / f).exists()]
X = pd.concat(frames, axis=1)
X = X.loc[:, ~X.columns.duplicated(keep="first")].astype("float32")
X = X.mask(np.isinf(X))
features = list(X.columns)

fields, cat = load_universe()
close = fields["Close"]
lbl = build_label_h(close, cat, HORIZON, SKIP)
d = X.join(lbl, how="inner").dropna(subset=["target"])
d = d.dropna(subset=features, how="all")
d["target"] = d["target"].astype("int8")
date_index = d.index.get_level_values("date")
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y_top"] = (day_pct >= 1.0 - TOP_PCT).astype("int8")
d["y_ext"] = ((day_pct >= 1.0 - TOP_PCT) | (day_pct <= TOP_PCT)).astype("int8")


def window_masks(years):
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
    test_end = pd.Timestamp(f"{TEST_YEAR}-12-31")
    cut = close.index[max(close.index.searchsorted(test_start) - EMBARGO, 0)]
    start = test_start - pd.DateOffset(years=years)
    tr = (date_index <= cut) & (date_index >= start)
    te = (date_index >= test_start) & (date_index <= test_end)
    tr_dates = np.sort(date_index[tr].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.90)]
    val_cut = close.index[max(close.index.searchsorted(val_start) - EMBARGO, 0)]
    fit = tr & (date_index <= val_cut)
    val = tr & (date_index >= val_start)
    return fit, val, te

fit_m, val_m, te_m = window_masks(1)
print(f"2-week + tuned hp, 1y window: fit={fit_m.sum()} val={val_m.sum()} "
     f"test={te_m.sum()}", flush=True)

from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, tree_method="hist", device="cuda",
           early_stopping_rounds=150, random_state=42, verbosity=0)

# stage2 (direction) with TUNED hyperparameters
ft, vt = fit_m & (d["y_ext"] == 1).values, val_m & (d["y_ext"] == 1).values
with gpu_lock():
    stage2 = XGBClassifier(**BASE, **TUNED, objective="binary:logistic",
                           eval_metric="auc")
    stage2.fit(d.loc[ft, features], d.loc[ft, "y_top"],
              eval_set=[(d.loc[vt, features], d.loc[vt, "y_top"])], verbose=False)

tt = te_m & (d["y_ext"] == 1).values
tail_auc = roc_auc_score(d.loc[tt, "y_top"], stage2.predict_proba(d.loc[tt, features])[:, 1])
print(f"stage2 (tuned, 2-week) tail_auc={tail_auc:.4f}  best_iter={stage2.best_iteration}")

# stage1 (extreme gate) keeps its own known-good default settings
y_ext_fit = d.loc[fit_m, "y_ext"]
spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
with gpu_lock():
    stage1 = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0,
                           objective="binary:logistic", scale_pos_weight=spw,
                           eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], y_ext_fit,
              eval_set=[(d.loc[val_m, features], d.loc[val_m, "y_ext"])], verbose=False)

p_ext = stage1.predict_proba(d.loc[te_m, features])[:, 1]
p_dir = stage2.predict_proba(d.loc[te_m, features])[:, 1]
y_te = d.loc[te_m, "target"].to_numpy()

is_ext = p_ext >= 0.5
pred_ens = np.where(~is_ext, 1, np.where(p_dir >= 0.5, 2, 0))
cm = confusion_matrix(y_te, pred_ens, labels=[0, 1, 2])
u2o, o2u = cm[0, 2] / cm[0].sum(), cm[2, 0] / cm[2].sum()
print(f"\nargmax ensemble: acc={accuracy_score(y_te, pred_ens):.4f}  "
     f"under->over={u2o:.3f}  over->under={o2u:.3f}")
print(pd.DataFrame(cm, index=MC_NAMES, columns=MC_NAMES).to_string())

COST_AGGRESSIVE = np.array([[0, 2, 6], [1, 0, 1], [6, 2, 0]], dtype=float)
P_ens = np.column_stack([p_ext * (1 - p_dir), 1 - p_ext, p_ext * p_dir])
cost_ud = np.minimum(P_ens @ COST_AGGRESSIVE[:, 0], P_ens @ COST_AGGRESSIVE[:, 2])
saving = (P_ens @ COST_AGGRESSIVE[:, 1]) - cost_ud
direction = np.where((P_ens @ COST_AGGRESSIVE[:, 2]) < (P_ens @ COST_AGGRESSIVE[:, 0]), 2, 0)
te_df = pd.DataFrame({"saving": saving, "direction": direction, "y_true": y_te,
                     "rel": d.loc[te_m, "rel"].to_numpy()}, index=d.index[te_m])

rows = []
for k in KS:
    picks = te_df.groupby(level="date").apply(
        lambda g: g.nlargest(max(int(len(g) * k), 1), "saving"))
    picks = picks.droplevel(0) if picks.index.nlevels > 2 else picks
    win = (picks["direction"] == picks["y_true"]).to_numpy()
    sign = np.where(picks["direction"] == 2, 1, -1)
    payoff = sign * picks["rel"].to_numpy()
    wins, losses = payoff[payoff > 0], payoff[payoff < 0]
    wl = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan
    rows.append({"top_k_pct": k, "n_calls": len(picks), "win_rate": round(float(win.mean()), 4),
                "mean_payoff": round(float(payoff.mean()), 5),
                "win_loss_ratio": round(wl, 3) if wl else None})
    print(f"K={k:>5.1%}  n={len(picks):>5}  win_rate={win.mean():.4f}  "
         f"payoff={payoff.mean():.5f}  wl_ratio={wl:.3f}", flush=True)

res = pd.DataFrame(rows)
print("\n===== combined (2-week + tuned hp) results =====")
print(res.to_string(index=False))
res.to_csv(EXP / "results" / "combined_best.csv", index=False)
print("\nsaved -> experiments/results/combined_best.csv")
