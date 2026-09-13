"""train_production — train and persist the production signal (XGB+GNN blend).

The winner of the eval-protocol comparison: blend of the two-stage XGBoost
ensemble (extreme gate x tails direction, 10% training threshold, tuned
stage2) and the CatGNN peer-structure model, blended per-day-z-scored with
the weight chosen by validation rank IC. 2-week horizon, 1y rolling window.

Persists everything predict_signal.py needs under experiments/production/:
  stage1.ubj, stage2.ubj      XGBoost boosters
  catgnn.pt                   GNN weights
  scaler.parquet              median/mean/std per feature (GNN preprocessing)
  config.json                 features, categories, blend weight, thresholds

Also runs the final blend through eval_protocol on 2026 and records the
scorecard (eval_production.json) so every retrain leaves an audit trail.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, datetime
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, load_universe, gpu_lock, EXP
from eval_protocol import evaluate
import nn_common as nc

HORIZON, WINDOW_YEARS, TRAIN_THR = 10, 1, 0.10
PROD = EXP / "production"
PROD.mkdir(exist_ok=True)

d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(
    horizon=HORIZON, window_years=WINDOW_YEARS)
print(f"dataset: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}",
      flush=True)
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
dates_all = d.index.get_level_values("date")
rel_all = d["rel"].to_numpy(dtype="float32")

# ---------------------------------------------------------------- XGB two-stage
from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, tree_method="hist", device="cuda",
            early_stopping_rounds=150, random_state=42, verbosity=0)
TUNED = dict(max_depth=6, learning_rate=0.03, min_child_weight=8,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)

y_top_tr = (day_pct >= 1 - TRAIN_THR).astype("int8")
y_ext_tr = ((day_pct >= 1 - TRAIN_THR) | (day_pct <= TRAIN_THR)).astype("int8")
y_ext_15 = ((day_pct >= 0.85) | (day_pct <= 0.15)).astype("int8")

ft, vt = fit_m & (y_ext_tr == 1).values, val_m & (y_ext_tr == 1).values
with gpu_lock():
    stage2 = XGBClassifier(**BASE, **TUNED, objective="binary:logistic",
                           eval_metric="auc")
    stage2.fit(d.loc[ft, features], y_top_tr[ft],
               eval_set=[(d.loc[vt, features], y_top_tr[vt])], verbose=False)
    print(f"stage2 trained (best_iter={stage2.best_iteration})", flush=True)

    y_ext_fit = y_ext_15[fit_m]
    spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
    stage1 = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0,
                           objective="binary:logistic", scale_pos_weight=spw,
                           eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], y_ext_fit,
               eval_set=[(d.loc[val_m, features], y_ext_15[val_m])],
               verbose=False)
    print(f"stage1 trained (best_iter={stage1.best_iteration})", flush=True)


def xgb_score(mask):
    p_ext = stage1.predict_proba(d.loc[mask, features])[:, 1]
    p_dir = stage2.predict_proba(d.loc[mask, features])[:, 1]
    return p_ext * (2.0 * p_dir - 1.0)

# ---------------------------------------------------------------- CatGNN
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

med, mu, sd = nc.standardize_stats(d, features, fit_m)
Xall = nc.standardize(d[features], med, mu, sd)
categories = sorted(pd.Series(
    d.index.get_level_values("ticker").map(load_universe()[1])).dropna()
    .unique().tolist())
cat_map = {c: i for i, c in enumerate(categories)}
cat_codes = np.array([cat_map.get(c, -1) for c in
                      d.index.get_level_values("ticker").map(
                          load_universe()[1])])
cat_codes[cat_codes < 0] = len(categories)          # unknowns -> own group

days_fit = nc.day_slices(fit_m, dates_all, cat_codes)
days_val = nc.day_slices(val_m, dates_all, cat_codes)
days_te = nc.day_slices(te_m, dates_all, cat_codes)

if device.type == "cuda":
    with gpu_lock():
        gnn, gnn_val_ic, gnn_eps = nc.train_catgnn(
            Xall, days_fit, days_val, rel_all, device)
else:
    gnn, gnn_val_ic, gnn_eps = nc.train_catgnn(
        Xall, days_fit, days_val, rel_all, device)
print(f"gnn trained: best val IC {gnn_val_ic:+.4f} ({gnn_eps} epochs)",
      flush=True)


def gnn_score(mask, days):
    return nc.day_scores(gnn, days, Xall, device)

# ---------------------------------------------------------------- blend weight
def daily_z(mask, s):
    ser = pd.Series(s, index=d.index[mask])
    g = ser.groupby(level="date")
    return ((ser - g.transform("mean")) /
            g.transform("std").replace(0, 1.0)).to_numpy()


def daily_ic(mask, s):
    df = pd.DataFrame({"s": s, "rel": d.loc[mask, "rel"]}, index=d.index[mask])
    return float(df.groupby(level="date")
                 .apply(lambda g: g["s"].corr(g["rel"], method="spearman"))
                 .mean())


sx_val = daily_z(val_m, xgb_score(val_m))
sg_val = daily_z(val_m, gnn_score(val_m, days_val))
best_w, best_ic = 0.5, -np.inf
for w in np.arange(0.0, 1.01, 0.1):
    ic = daily_ic(val_m, w * sx_val + (1 - w) * sg_val)
    if ic > best_ic:
        best_w, best_ic = float(w), ic
print(f"blend weight w_xgb={best_w:.1f} (val IC {best_ic:+.4f})", flush=True)

# ---------------------------------------------------------------- persist
stage1.get_booster().save_model(str(PROD / "stage1.ubj"))
stage2.get_booster().save_model(str(PROD / "stage2.ubj"))
torch.save(gnn.state_dict(), PROD / "catgnn.pt")
pd.DataFrame({"median": med, "mean": mu, "std": sd}).to_parquet(
    PROD / "scaler.parquet")
config = {
    "trained": datetime.datetime.now().isoformat(timespec="seconds"),
    "horizon": HORIZON, "window_years": WINDOW_YEARS,
    "train_extreme_thr": TRAIN_THR, "w_xgb": best_w,
    "gnn_hidden": nc.HID, "features": features, "categories": categories,
    "gnn_val_ic": round(gnn_val_ic, 4), "blend_val_ic": round(best_ic, 4),
}
(PROD / "config.json").write_text(json.dumps(config, indent=2))
print("artifacts saved ->", PROD, flush=True)

# ---------------------------------------------------------------- 2026 audit
sx_te = daily_z(te_m, xgb_score(te_m))
sg_te = daily_z(te_m, gnn_score(te_m, days_te))
s_blend = best_w * sx_te + (1 - best_w) * sg_te
res = evaluate("production", d, te_m, s_blend,
               extra={"w_xgb": best_w, "horizon": HORIZON,
                      "window_years": WINDOW_YEARS})
print(f"\nmacro F1 (terciles): {res['macro_f1_terciles']}")
