"""fix_direction — compare 3 ways to fix the under/over confusion problem.

Diagnosis (confusion_report.py, full-history models): every multiclass model
gets "neutral" right often (60-65% recall) but confuses under<->over with
each other almost as often as it gets them right (37-45% cross vs 40-45%
correct) -- it detects EXTREMENESS fine but not DIRECTION.

All three below use the 1y training window (best-validated config from
validate_window.py: multiclass rank IC 0.031, ensemble rank IC 0.071 on
2026, vs 0.026 / 0.056 for full history) so the comparison isn't muddied by
window-size effects on top of the fix itself.

  A. baseline  — plain 3-class XGBoost (today's approach, for comparison)
  B. ensemble  — stage1 extreme-vs-neutral (balanced) -> stage2 top-vs-bottom
                 direction (only on stage1's "extreme" calls); bucketed back
                 to 3 classes so its confusion matrix is directly comparable
  C. ordinal   — XGBoost regressor on label {0,1,2} with squared error, so a
                 distance-2 miss (under<->over) costs 4x a distance-1 miss
                 (either<->neutral) by construction; bucketed at 0.5/1.5
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from exp_harness import load_sweep_dataset, EMBARGO, gpu_lock, EXP

TEST_YEAR, TOP_PCT = 2026, 0.15
MC_NAMES = ["under", "neutral", "over"]

d, features, _, _, _, trading_index = load_sweep_dataset()
date_index = d.index.get_level_values("date")

day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y_top"] = (day_pct >= 1.0 - TOP_PCT).astype("int8")
d["y_ext"] = ((day_pct >= 1.0 - TOP_PCT) | (day_pct <= TOP_PCT)).astype("int8")

# ---------------------------------------------------------------- 1y window
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

y_te = d.loc[te_m, "target"].to_numpy()
from xgboost import XGBClassifier, XGBRegressor

BASE = dict(n_estimators=3000, learning_rate=0.03, max_depth=8,
            min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, tree_method="hist", device="cuda",
            early_stopping_rounds=150, random_state=42, verbosity=0)

results = {}

def report(name, pred, key):
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    acc = accuracy_score(y_te, pred)
    mf1 = f1_score(y_te, pred, average="macro")
    # under<->over confusion rate: of true extremes, how often flipped sign
    under_as_over = cm[0, 2] / cm[0].sum()
    over_as_under = cm[2, 0] / cm[2].sum()
    print(f"\n================ {name}  (acc={acc:.4f}, macro_f1={mf1:.4f}, "
          f"under->over={under_as_over:.3f}, over->under={over_as_under:.3f})")
    print(pd.DataFrame(cm, index=MC_NAMES, columns=MC_NAMES).to_string())
    print("row-normalized (recall):")
    print(pd.DataFrame(cmn.round(3), index=MC_NAMES, columns=MC_NAMES).to_string())
    results[key] = {"acc": round(float(acc), 4), "macro_f1": round(float(mf1), 4),
                    "under_to_over": round(float(under_as_over), 4),
                    "over_to_under": round(float(over_as_under), 4)}

# ---------------------------------------------------------------- A. baseline
with gpu_lock():
    base = XGBClassifier(**BASE, objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss")
    base.fit(d.loc[fit_m, features], d.loc[fit_m, "target"],
             eval_set=[(d.loc[val_m, features], d.loc[val_m, "target"])],
             verbose=False)
pred_base = base.predict_proba(d.loc[te_m, features]).argmax(1)
report("A. Baseline multiclass (1y window)", pred_base, "baseline")

# ---------------------------------------------------------------- B. ensemble
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
is_ext = p_ext >= 0.5
pred_ens = np.where(~is_ext, 1, np.where(p_dir >= 0.5, 2, 0))
report("B. Two-stage ensemble (extreme-gate x direction), bucketed", pred_ens, "ensemble")

# rank IC for the ensemble's continuous score (not just the bucketed class)
s_ens = p_ext * (2.0 * p_dir - 1.0)
te_df = pd.DataFrame({"s": s_ens, "rel": d.loc[te_m, "rel"]}, index=d.index[te_m])
ic_ens = te_df.groupby(level="date").apply(
    lambda g: g["s"].corr(g["rel"], method="spearman")).mean()
results["ensemble"]["rank_ic"] = round(float(ic_ens), 4)

# ---------------------------------------------------------------- C. ordinal
with gpu_lock():
    ordm = XGBRegressor(**BASE, objective="reg:squarederror", eval_metric="rmse")
    ordm.fit(d.loc[fit_m, features], d.loc[fit_m, "target"].astype(float),
             eval_set=[(d.loc[val_m, features], d.loc[val_m, "target"].astype(float))],
             verbose=False)
raw = ordm.predict(d.loc[te_m, features])
# fixed 0.5/1.5 rounding collapses to all-neutral (raw scores cluster tightly
# around 1.0) -- bucket by daily percentile instead, same tercile convention
# used to build the label itself, so classes stay balanced like the others
raw_pct = pd.Series(raw, index=d.index[te_m]).groupby(level="date").rank(pct=True)
pred_ord = np.select([raw_pct <= 1 / 3, raw_pct <= 2 / 3], [0, 1], default=2)
report("C. Ordinal regression (squared-error, percentile-bucketed)", pred_ord, "ordinal")

te_df2 = pd.DataFrame({"s": raw, "rel": d.loc[te_m, "rel"]}, index=d.index[te_m])
ic_ord = te_df2.groupby(level="date").apply(
    lambda g: g["s"].corr(g["rel"], method="spearman")).mean()
results["ordinal"]["rank_ic"] = round(float(ic_ord), 4)

# baseline rank IC too, for fair comparison
proba_base = base.predict_proba(d.loc[te_m, features])
te_df3 = pd.DataFrame({"s": proba_base[:, 2], "rel": d.loc[te_m, "rel"]},
                      index=d.index[te_m])
ic_base = te_df3.groupby(level="date").apply(
    lambda g: g["s"].corr(g["rel"], method="spearman")).mean()
results["baseline"]["rank_ic"] = round(float(ic_base), 4)

print("\n\n===== summary (2026 holdout, 1y training window) =====")
print(pd.DataFrame(results).T.to_string())

import json
(EXP / "results" / "fix_direction.json").write_text(json.dumps(results, indent=2))
print("\nsaved -> experiments/results/fix_direction.json")
