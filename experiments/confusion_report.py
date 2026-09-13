"""confusion_report — confusion matrices for every model trained so far (2026 test).

CatBoost and the LSTM are loaded from their saved artifacts; XGBoost /
LightGBM / logistic / the two binary models were not persisted, so they are
refit with identical hyperparameters, seeds, and masks as their original
scripts (metrics are checked against the stored jsons to confirm the refit
reproduces the original model). Test predictions are saved this time
(results/preds_2026.parquet) so future questions don't need refits.

Multiclass label: 0 = underperform, 1 = neutral, 2 = outperform.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from exp_harness import (load_sweep_dataset, load_universe, loo_peer_mean,
                         cat_pct_rank, build_label, holdout_masks, gpu_lock, EXP)

MC_NAMES = ["under", "neutral", "over"]
TOP_PCT = 0.15
out_lines, preds_store = [], {}

def report(name, y_true, y_pred, labels, check=None):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    lines = [f"\n================ {name}  (acc={acc:.4f}, macro_f1={mf1:.4f})",
             "counts (rows=true, cols=pred):",
             pd.DataFrame(cm, index=labels, columns=labels).to_string(),
             "row-normalized (recall per true class):",
             pd.DataFrame(cmn.round(3), index=labels, columns=labels).to_string()]
    if check is not None:
        lines.append(f"reproduction check vs stored json: refit f1={mf1:.4f} "
                     f"vs original {check} "
                     f"({'OK' if abs(mf1 - check) < 0.01 else 'MISMATCH'})")
    txt = "\n".join(lines)
    print(txt, flush=True)
    out_lines.append(txt)

d, features, fit_mask, val_mask, te_mask, trading_index = load_sweep_dataset()
y_te = d.loc[te_mask, "target"].to_numpy()
te_index = d.index[te_mask]
stored = {p.stem: json.loads(p.read_text())
          for p in (EXP / "results").glob("model_*.json")}

# ---------------------------------------------------------------- catboost (refit)
# (the saved .cbm is from the with-composition 96-feature rerun, so it can't
# score the canonical 74-feature matrix -- refit like the others instead)
from catboost import CatBoostClassifier
with gpu_lock():
    cb = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=10, l2_leaf_reg=9,
        loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
        random_seed=42, task_type="GPU", verbose=False,
        early_stopping_rounds=150, use_best_model=True)
    cb.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
           eval_set=(d.loc[val_mask, features], d.loc[val_mask, "target"]))
pred = np.asarray(cb.predict(d.loc[te_mask, features])).ravel().astype(int)
preds_store["catboost"] = pred
report("CatBoost (refit)", y_te, pred, MC_NAMES,
       check=stored["model_catboost"]["macro_f1"])

# ---------------------------------------------------------------- xgboost (refit)
from xgboost import XGBClassifier
with gpu_lock():
    xgb = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="multi:softprob", num_class=3,
        tree_method="hist", device="cuda", eval_metric="mlogloss",
        early_stopping_rounds=150, random_state=42, verbosity=0)
    xgb.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
            eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "target"])],
            verbose=False)
pred = xgb.predict_proba(d.loc[te_mask, features]).argmax(1)
preds_store["xgboost"] = pred
report("XGBoost (refit)", y_te, pred, MC_NAMES,
       check=stored["model_xgboost"]["macro_f1"])

# ---------------------------------------------------------------- lightgbm (refit)
import lightgbm as lgb
lg = lgb.LGBMClassifier(
    n_estimators=3000, learning_rate=0.03, num_leaves=255, max_depth=-1,
    min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=5.0, objective="multiclass",
    num_class=3, n_jobs=12, random_state=42, verbosity=-1)
lg.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
       eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "target"])],
       eval_metric="multi_logloss", callbacks=[lgb.early_stopping(150, verbose=False)])
pred = lg.predict_proba(d.loc[te_mask, features]).argmax(1)
preds_store["lightgbm"] = pred
report("LightGBM (refit)", y_te, pred, MC_NAMES,
       check=stored["model_lightgbm"]["macro_f1"])

# ---------------------------------------------------------------- logreg (refit)
tr_mask = fit_mask | val_mask
med = d.loc[tr_mask, features].median()
mu = d.loc[tr_mask, features].mean()
sd = d.loc[tr_mask, features].std().replace(0, 1.0)
def prep(mask):
    Z = (d.loc[mask, features].fillna(med) - mu) / sd
    return Z.clip(-5, 5).to_numpy(dtype="float32")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=300, C=1.0, tol=1e-3)
lr.fit(prep(tr_mask), d.loc[tr_mask, "target"])
pred = lr.predict(prep(te_mask))
preds_store["logreg"] = pred
report("Logistic regression (refit)", y_te, pred, MC_NAMES,
       check=stored["model_logreg"]["macro_f1"])

# ---------------------------------------------------------------- binary top15 (refit)
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
d["y_top"] = (day_pct >= 1.0 - TOP_PCT).astype("int8")
d["y_ext"] = ((day_pct >= 1.0 - TOP_PCT) | (day_pct <= TOP_PCT)).astype("int8")
y_fit = d.loc[fit_mask, "y_top"]
spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
with gpu_lock():
    b1 = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="binary:logistic", scale_pos_weight=spw,
        tree_method="hist", device="cuda", eval_metric="aucpr",
        early_stopping_rounds=150, random_state=42, verbosity=0)
    b1.fit(d.loc[fit_mask, features], y_fit,
           eval_set=[(d.loc[val_mask, features], d.loc[val_mask, "y_top"])],
           verbose=False)
pred = (b1.predict_proba(d.loc[te_mask, features])[:, 1] >= 0.5).astype(int)
preds_store["binary_top15"] = pred
report("Binary top-15%-vs-rest (refit, thresh 0.5)",
       d.loc[te_mask, "y_top"].to_numpy(), pred, ["rest", "top15"])

# ---------------------------------------------------------------- binary tails (refit)
ft, vt = fit_mask & (d["y_ext"] == 1).values, val_mask & (d["y_ext"] == 1).values
tt = te_mask & (d["y_ext"] == 1).values
with gpu_lock():
    b2 = XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, objective="binary:logistic",
        tree_method="hist", device="cuda", eval_metric="auc",
        early_stopping_rounds=150, random_state=42, verbosity=0)
    b2.fit(d.loc[ft, features], d.loc[ft, "y_top"],
           eval_set=[(d.loc[vt, features], d.loc[vt, "y_top"])], verbose=False)
pred = (b2.predict_proba(d.loc[tt, features])[:, 1] >= 0.5).astype(int)
report("Binary top15-vs-bottom15 tails (refit, thresh 0.5, tail rows only)",
       d.loc[tt, "y_top"].to_numpy(), pred, ["bottom15", "top15"])

# save tabular test predictions for future use
pd.DataFrame(preds_store, index=te_index).to_parquet(
    EXP / "results" / "preds_2026.parquet")

# ---------------------------------------------------------------- lstm (saved .pt)
# rebuild channels/samples exactly as model_lstm.py, then load weights
SEQ, N_CH = 20, 8
fields, cat = load_universe()
o, h, l, c, v = (fields[k] for k in ["Open", "High", "Low", "Close", "Volume"])
r1 = c.pct_change(fill_method=None)
logv = np.log1p(v.where(v > 0))
hl_rng = h - l
channels = {
    "r1": r1, "r1_rel": r1 - loo_peer_mean(r1, cat),
    "rank_r1": cat_pct_rank(r1, cat) - 0.5, "hl": hl_rng / c,
    "clv": ((c - l) - (h - c)) / hl_rng.where(hl_rng > 0),
    "gap": o / c.shift(1) - 1.0,
    "volz": (logv - logv.rolling(60).mean()) / logv.rolling(60).std(),
    "dd60": c / c.rolling(60).max() - 1.0,
}
C = np.stack([ch.to_numpy(dtype="float32") for ch in channels.values()], axis=-1)
dates, tickers = c.index, c.columns
lbl = build_label(c, cat)
samp = lbl.reset_index()
samp["di"] = dates.get_indexer(samp["date"])
samp["ti"] = pd.Index(tickers).get_indexer(samp["ticker"])
samp = samp[(samp["di"] >= SEQ - 1) & (samp["ti"] >= 0)]
samp = samp.set_index(["date", "ticker"])
date_index = samp.index.get_level_values("date")
fit_m, val_m, te_m = holdout_masks(date_index, dates)
nonoverlap = date_index.isin(dates[::5])
fit_m = fit_m & nonoverlap
fit_cut = date_index[fit_m].max()
Cfit = C[: dates.searchsorted(fit_cut) + 1]
mu_c = np.nanmean(Cfit, axis=(0, 1)); sd_c = np.nanstd(Cfit, axis=(0, 1))
sd_c[sd_c == 0] = 1.0
Cn = np.nan_to_num(np.clip((C - mu_c) / sd_c, -5, 5), nan=0.0).astype("float32")
del C, Cfit

import torch, torch.nn as nn
class SeqNet(nn.Module):
    def __init__(self, n_ch=N_CH, hidden=96):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, num_layers=2, batch_first=True,
                            dropout=0.25)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.25),
                                  nn.Linear(hidden, 3))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeqNet().to(device)
model.load_state_dict(torch.load(EXP / "results" / "model_lstm.pt",
                                 map_location=device, weights_only=True))
model.eval()
ar = np.arange(SEQ) - (SEQ - 1)
di = samp.loc[te_m, "di"].to_numpy(); ti = samp.loc[te_m, "ti"].to_numpy()
preds = []
with torch.no_grad():
    for s in range(0, len(di), 8192):
        sl = slice(s, min(s + 8192, len(di)))
        xb = torch.from_numpy(Cn[di[sl, None] + ar[None, :], ti[sl, None], :]).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
pred = np.concatenate(preds)
report("LSTM (loaded from .pt)", samp.loc[te_m, "target"].to_numpy().astype(int),
       pred, MC_NAMES, check=stored["model_lstm"]["macro_f1"])

(EXP / "results" / "confusion_matrices.txt").write_text("\n".join(out_lines),
                                                        encoding="utf-8")
print("\nsaved -> experiments/results/confusion_matrices.txt "
      "and preds_2026.parquet")
