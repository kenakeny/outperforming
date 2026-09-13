"""hp_tune_direction — hyperparameter search for the direction (tails) model.

Every ensemble run so far reused the same default XGBoost config borrowed
from the original multiclass sweep. Nobody has actually searched for good
hyperparameters on the DIRECTION sub-problem specifically. Stage1 (extreme
gate) is trained once with known-good settings and reused across the whole
grid so compute goes entirely into varying stage2 (direction).

1y training window (best-validated), 2026 test only, per user instruction
to focus on 2026. For each stage2 config: direction AUC/accuracy on the
tail-only test rows, plus win_rate/payoff at a K=3%/day confidence gate
(the sweet spot found in selectivity_sweep.py) so the comparison reflects
what actually gets deployed, not just a proxy metric.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from exp_harness import load_sweep_dataset, EMBARGO, gpu_lock, EXP

TEST_YEAR, TOP_PCT, GATE_K = 2026, 0.15, 0.03
COST_AGGRESSIVE = np.array([[0, 2, 6], [1, 0, 1], [6, 2, 0]], dtype=float)

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

STAGE1_PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=8,
                     min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=5.0, tree_method="hist", device="cuda",
                     early_stopping_rounds=150, random_state=42, verbosity=0)

y_ext_fit = d.loc[fit_m, "y_ext"]
spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
with gpu_lock():
    stage1 = XGBClassifier(**STAGE1_PARAMS, objective="binary:logistic",
                           scale_pos_weight=spw, eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], y_ext_fit,
              eval_set=[(d.loc[val_m, features], d.loc[val_m, "y_ext"])], verbose=False)
p_ext = stage1.predict_proba(d.loc[te_m, features])[:, 1]
print("stage1 (extreme gate) trained once, reused for every config", flush=True)

ft, vt = fit_m & (d["y_ext"] == 1).values, val_m & (d["y_ext"] == 1).values
tt = te_m & (d["y_ext"] == 1).values

GRID = [
    {"depth": 8,  "lr": 0.03, "mcw": 8,  "sub": 0.8, "col": 0.8, "l2": 5},   # current default
    {"depth": 4,  "lr": 0.03, "mcw": 8,  "sub": 0.8, "col": 0.8, "l2": 5},
    {"depth": 6,  "lr": 0.03, "mcw": 8,  "sub": 0.8, "col": 0.8, "l2": 5},
    {"depth": 4,  "lr": 0.01, "mcw": 20, "sub": 0.7, "col": 0.7, "l2": 10},
    {"depth": 5,  "lr": 0.02, "mcw": 20, "sub": 0.7, "col": 0.7, "l2": 10},
    {"depth": 4,  "lr": 0.01, "mcw": 40, "sub": 0.6, "col": 0.6, "l2": 20},
    {"depth": 3,  "lr": 0.02, "mcw": 30, "sub": 0.7, "col": 0.7, "l2": 15},
    {"depth": 6,  "lr": 0.01, "mcw": 20, "sub": 0.6, "col": 0.6, "l2": 15},
    {"depth": 4,  "lr": 0.005,"mcw": 50, "sub": 0.6, "col": 0.6, "l2": 30},
    {"depth": 2,  "lr": 0.03, "mcw": 20, "sub": 0.8, "col": 0.8, "l2": 10},
]

rows = []
for i, cfg in enumerate(GRID):
    with gpu_lock():
        m = XGBClassifier(
            n_estimators=3000, learning_rate=cfg["lr"], max_depth=cfg["depth"],
            min_child_weight=cfg["mcw"], subsample=cfg["sub"],
            colsample_bytree=cfg["col"], reg_lambda=cfg["l2"],
            objective="binary:logistic", tree_method="hist", device="cuda",
            eval_metric="auc", early_stopping_rounds=150,
            random_state=42, verbosity=0)
        m.fit(d.loc[ft, features], d.loc[ft, "y_top"],
              eval_set=[(d.loc[vt, features], d.loc[vt, "y_top"])], verbose=False)
    p_dir_tail = m.predict_proba(d.loc[tt, features])[:, 1]
    auc = roc_auc_score(d.loc[tt, "y_top"], p_dir_tail)
    acc = accuracy_score(d.loc[tt, "y_top"], (p_dir_tail >= 0.5).astype(int))

    p_dir_all = m.predict_proba(d.loc[te_m, features])[:, 1]
    P_ens = np.column_stack([p_ext * (1 - p_dir_all), 1 - p_ext, p_ext * p_dir_all])
    cost_ud = np.minimum(P_ens @ COST_AGGRESSIVE[:, 0], P_ens @ COST_AGGRESSIVE[:, 2])
    saving = (P_ens @ COST_AGGRESSIVE[:, 1]) - cost_ud
    direction = np.where((P_ens @ COST_AGGRESSIVE[:, 2]) < (P_ens @ COST_AGGRESSIVE[:, 0]), 2, 0)
    te_df = pd.DataFrame({"saving": saving, "direction": direction,
                          "y_true": d.loc[te_m, "target"].to_numpy(),
                          "rel": d.loc[te_m, "rel"].to_numpy()}, index=d.index[te_m])
    picks = te_df.groupby(level="date").apply(
        lambda g: g.nlargest(max(int(len(g) * GATE_K), 1), "saving"))
    picks = picks.droplevel(0) if picks.index.nlevels > 2 else picks
    win_rate = float((picks["direction"] == picks["y_true"]).mean())
    sign = np.where(picks["direction"] == 2, 1, -1)
    payoff = float((sign * picks["rel"]).mean())
    best_iter = int(m.best_iteration)

    row = {"cfg": i, **cfg, "best_iter": best_iter, "tail_auc": round(float(auc), 4),
          "tail_acc": round(float(acc), 4), "gated_win_rate": round(win_rate, 4),
          "gated_payoff": round(payoff, 5)}
    rows.append(row)
    print(f"[{i}] depth={cfg['depth']} lr={cfg['lr']} mcw={cfg['mcw']} "
         f"sub={cfg['sub']} l2={cfg['l2']}  ->  tail_auc={auc:.4f}  "
         f"tail_acc={acc:.4f}  gated_win_rate={win_rate:.4f}  "
         f"gated_payoff={payoff:.5f}  iters={best_iter}", flush=True)

res = pd.DataFrame(rows).sort_values("tail_auc", ascending=False)
print("\n===== hyperparameter sweep results (sorted by tail_auc) =====")
print(res.to_string(index=False))
res.to_csv(EXP / "results" / "hp_tune_direction.csv", index=False)
print("\nsaved -> experiments/results/hp_tune_direction.csv")
