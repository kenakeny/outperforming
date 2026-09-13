
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from exp_harness import load_universe, gpu_lock, EXP, SWEEP_FAMILIES

HORIZON, SKIP = 10, 0
EMBARGO = HORIZON + SKIP
TEST_YEAR = 2026
TRAIN_EXTREME_THR = 0.10          # winning threshold from fix_flip_hard.py
EVAL_EXTREME_THR = 0.15           # eval population stays the standard 15%
CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70]
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

d["y_top_eval"] = (day_pct >= 1.0 - EVAL_EXTREME_THR).astype("int8")
d["y_ext_eval"] = ((day_pct >= 1.0 - EVAL_EXTREME_THR) |
                   (day_pct <= EVAL_EXTREME_THR)).astype("int8")
y_top_tr = (day_pct >= 1.0 - TRAIN_EXTREME_THR).astype("int8")
y_ext_tr = ((day_pct >= 1.0 - TRAIN_EXTREME_THR) |
           (day_pct <= TRAIN_EXTREME_THR)).astype("int8")


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
print(f"1y window: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}", flush=True)

from xgboost import XGBClassifier
BASE = dict(n_estimators=3000, tree_method="hist", device="cuda",
           early_stopping_rounds=150, random_state=42, verbosity=0)

# stage2: direction, tuned hyperparams, 10% extreme training threshold
ft, vt = fit_m & (y_ext_tr == 1).values, val_m & (y_ext_tr == 1).values
with gpu_lock():
    stage2 = XGBClassifier(**BASE, **TUNED, objective="binary:logistic",
                           eval_metric="auc")
    stage2.fit(d.loc[ft, features], y_top_tr[ft],
              eval_set=[(d.loc[vt, features], y_top_tr[vt])], verbose=False)

# stage1: extreme gate, same settings as every prior run
y_ext_fit = d.loc[fit_m, "y_ext_eval"]
spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
with gpu_lock():
    stage1 = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0,
                           objective="binary:logistic", scale_pos_weight=spw,
                           eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], y_ext_fit,
              eval_set=[(d.loc[val_m, features], d.loc[val_m, "y_ext_eval"])],
              verbose=False)

p_ext = stage1.predict_proba(d.loc[te_m, features])[:, 1]
p_dir = stage2.predict_proba(d.loc[te_m, features])[:, 1]
y_te = d.loc[te_m, "target"].to_numpy()

# ---------------------------------------------------------------- reference: plain baseline
with gpu_lock():
    base = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                         min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=5.0, objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss")
    base.fit(d.loc[fit_m, features], d.loc[fit_m, "target"],
            eval_set=[(d.loc[val_m, features], d.loc[val_m, "target"])], verbose=False)
pred_base = base.predict_proba(d.loc[te_m, features]).argmax(1)


def show(name, pred):
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    u2o, o2u = cm[0, 2] / cm[0].sum(), cm[2, 0] / cm[2].sum()
    print(f"\n================ {name}")
    print(f"under->over={u2o:.3f}  over->under={o2u:.3f}  "
         f"acc={accuracy_score(y_te, pred):.4f}  "
         f"macro_f1={f1_score(y_te, pred, average='macro'):.4f}")
    print(pd.DataFrame(cm, index=MC_NAMES, columns=MC_NAMES).to_string())
    print("row-normalized (recall):")
    print(pd.DataFrame(cmn.round(3), index=MC_NAMES, columns=MC_NAMES).to_string())

show("0. Plain baseline multiclass (2-week horizon, reference)", pred_base)

pred_ens_050 = np.where(p_ext < 0.5, 1, np.where(p_dir >= 0.5, 2, 0))
show("1. Ensemble, default 0.5 cutoff (today's config)", pred_ens_050)

for cutoff in CUTOFFS[1:]:
    lo, hi = 1.0 - cutoff, cutoff
    pred = np.where(p_ext < 0.5, 1,
                    np.where(p_dir >= hi, 2, np.where(p_dir <= lo, 0, 1)))
    show(f"Ensemble, direction cutoff {cutoff:.0%} (neutral if unsure)", pred)

print("\ndone")