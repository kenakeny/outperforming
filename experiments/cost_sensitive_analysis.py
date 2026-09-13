"""cost_sensitive_analysis — Bayes-optimal cost-sensitive decisions, baseline vs ensemble.

Follow-up to fix_direction.py. That script measured raw under<->over
confusion RATES at a fixed 0.5 decision threshold -- informative but not a
real cost analysis. Here we:

  1. Define explicit cost matrices C[true][pred] (0 on the diagonal).
  2. Get a full 3-class probability estimate from each model:
       - baseline: native softmax P(under), P(neutral), P(over)
       - ensemble: reconstructed from the two stages --
           P(neutral)  = 1 - P(extreme)
           P(over)     = P(extreme) * P(top | extreme)
           P(under)    = P(extreme) * (1 - P(top | extreme))
  3. Apply the Bayes-optimal cost-sensitive rule per row:
       pred = argmin_c  sum_k P(true=k) * C[k, c]
     This needs no threshold tuning/grid search -- it falls straight out of
     the cost matrix and the model's own probabilities, so there is no
     train/val fitting step and no leakage risk.
  4. Compare total expected cost, confusion matrices, and the specific
     under<->over flip rate against plain argmax for both models.

Same 1y training window (best-validated config), same 2026 test holdout,
same hyperparameters/seed as fix_direction.py, so results are a fair diff of
"same model, cost-aware decision rule instead of argmax."
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from exp_harness import load_sweep_dataset, EMBARGO, gpu_lock, EXP

TEST_YEAR, TOP_PCT = 2026, 0.15
MC_NAMES = ["under", "neutral", "over"]

COST_QUAD = np.array([[0, 1, 4],
                      [1, 0, 1],
                      [4, 1, 0]], dtype=float)          # (true-pred)^2

COST_AGGRESSIVE = np.array([[0, 2, 6],
                            [1, 0, 1],
                            [6, 2, 0]], dtype=float)     # flip=6, false-alarm=2, miss=1

COST_MATS = {"quadratic": COST_QUAD, "aggressive": COST_AGGRESSIVE}

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
y_te = d.loc[te_m, "target"].to_numpy()
print(f"1y window: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}", flush=True)

from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, learning_rate=0.03, max_depth=8,
            min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, tree_method="hist", device="cuda",
            early_stopping_rounds=150, random_state=42, verbosity=0)

# ---------------------------------------------------------------- fit models
with gpu_lock():
    base = XGBClassifier(**BASE, objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss")
    base.fit(d.loc[fit_m, features], d.loc[fit_m, "target"],
             eval_set=[(d.loc[val_m, features], d.loc[val_m, "target"])],
             verbose=False)
P_base = base.predict_proba(d.loc[te_m, features])          # [n, 3] under/neutral/over

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
p_dir = stage2.predict_proba(d.loc[te_m, features])[:, 1]   # P(top | extreme)
P_ens = np.column_stack([p_ext * (1 - p_dir),                # under
                         1 - p_ext,                          # neutral
                         p_ext * p_dir])                     # over

# ---------------------------------------------------------------- decisions
def bayes_decide(P, cost):
    """argmin_c sum_k P[:,k] * cost[k,c], vectorized over rows."""
    expected_cost = P @ cost                                  # [n, 3]
    return expected_cost.argmin(axis=1), expected_cost.min(axis=1)


def report(name, pred, exp_cost_mean=None):
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    acc = accuracy_score(y_te, pred)
    mf1 = f1_score(y_te, pred, average="macro")
    under_to_over = cm[0, 2] / cm[0].sum()
    over_to_under = cm[2, 0] / cm[2].sum()
    actual_cost = cost[y_te, pred].mean()  # realized cost using whichever matrix is in scope
    line = (f"\n================ {name}\n"
           f"acc={acc:.4f}  macro_f1={mf1:.4f}  "
           f"under->over={under_to_over:.3f}  over->under={over_to_under:.3f}  "
           f"realized_mean_cost={actual_cost:.4f}")
    print(line)
    print(pd.DataFrame(cm, index=MC_NAMES, columns=MC_NAMES).to_string())
    print("row-normalized (recall):")
    print(pd.DataFrame(cmn.round(3), index=MC_NAMES, columns=MC_NAMES).to_string())
    return {"acc": round(float(acc), 4), "macro_f1": round(float(mf1), 4),
           "under_to_over": round(float(under_to_over), 4),
           "over_to_under": round(float(over_to_under), 4),
           "realized_mean_cost": round(float(actual_cost), 4),
           "confusion_matrix": cm.tolist()}

results = {}

# argmax baselines (cost-agnostic, for reference)
pred_base_argmax = P_base.argmax(axis=1)
pred_ens_argmax = P_ens.argmax(axis=1)
for cost_name, cost in COST_MATS.items():
    results[f"baseline_argmax__{cost_name}cost"] = report(
        f"Baseline argmax, scored under {cost_name} cost", pred_base_argmax)
    results[f"ensemble_argmax__{cost_name}cost"] = report(
        f"Ensemble argmax (P reconstructed), scored under {cost_name} cost", pred_ens_argmax)

# Bayes-optimal decisions per cost matrix
for cost_name, cost in COST_MATS.items():
    print(f"\n\n########## COST MATRIX: {cost_name} ##########")
    print(pd.DataFrame(cost, index=MC_NAMES, columns=MC_NAMES).to_string())

    pred_base_cs, _ = bayes_decide(P_base, cost)
    results[f"baseline_costopt__{cost_name}"] = report(
        f"Baseline, Bayes-optimal decision ({cost_name} cost)", pred_base_cs)

    pred_ens_cs, _ = bayes_decide(P_ens, cost)
    results[f"ensemble_costopt__{cost_name}"] = report(
        f"Ensemble, Bayes-optimal decision ({cost_name} cost)", pred_ens_cs)

print("\n\n===== realized mean cost summary (lower is better; each row scored "
     "under its own matrix) =====")
summary = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                        for k, v in results.items()}).T
print(summary[["realized_mean_cost", "under_to_over", "over_to_under", "macro_f1"]]
     .to_string())

# ---------------------------------------------------------------- gated version
# The unconditional Bayes rule collapses to "always predict neutral" once the
# flip penalty dominates -- correct given how diffuse the probabilities are,
# but not actionable. Practical fix: only take a directional call on the
# subset where doing so actually beats the neutral default by the most (top
# 15%/day, same position-sizing convention as everywhere else in this repo),
# and score whether THOSE specific calls pay off.
print("\n\n########## confidence-gated cost-sensitive calls (top 15%/day) ##########")

def gated_report(name, P):
    cost_ud = np.minimum(P @ COST_AGGRESSIVE[:, 0], P @ COST_AGGRESSIVE[:, 2])
    cost_neutral = P @ COST_AGGRESSIVE[:, 1]
    saving = cost_neutral - cost_ud          # how much better than doing nothing
    direction = np.where((P @ COST_AGGRESSIVE[:, 2]) < (P @ COST_AGGRESSIVE[:, 0]), 2, 0)

    te = pd.DataFrame({"saving": saving, "direction": direction,
                       "y_true": y_te, "rel": d.loc[te_m, "rel"].to_numpy()},
                      index=d.index[te_m])
    picks = te.groupby(level="date").apply(
        lambda g: g.nlargest(max(int(len(g) * TOP_PCT), 1), "saving"))
    picks = picks.droplevel(0) if picks.index.nlevels > 2 else picks

    dir_correct = (picks["direction"] == picks["y_true"]).mean()
    sign = np.where(picks["direction"] == 2, 1, -1)
    payoff = float((sign * picks["rel"]).mean())
    called_neutral_truth = (picks["y_true"] == 1).mean()
    print(f"{name}: n_calls={len(picks)}  direction_correct={dir_correct:.4f}  "
         f"mean_signed_payoff={payoff:.5f}  fraction_actually_neutral={called_neutral_truth:.3f}")
    return {"n_calls": int(len(picks)), "direction_correct": round(float(dir_correct), 4),
           "mean_signed_payoff": round(payoff, 5),
           "fraction_actually_neutral": round(float(called_neutral_truth), 4)}

results["gated_baseline"] = gated_report("Baseline (gated, aggressive cost)", P_base)
results["gated_ensemble"] = gated_report("Ensemble (gated, aggressive cost)", P_ens)

(EXP / "results" / "cost_sensitive_analysis.json").write_text(
    json.dumps(results, indent=2))
print("\nsaved -> experiments/results/cost_sensitive_analysis.json")
