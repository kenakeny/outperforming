"""model_blend — blend the XGB two-stage ensemble score with the CatGNN score.

The GNN's first run didn't beat the benchmark head-on (IC 0.070 vs 0.082)
but its error profile is clearly different (lower flip rates, much higher
win/loss ratio, lower win rate) -- the signature of a decorrelated signal.
Standard move: blend. Both scores are z-scored per day, combined as
w * xgb + (1 - w) * gnn; the weight w is chosen by validation rank IC only
(no test peeking), then the chosen blend is evaluated once on 2026 via the
fixed protocol.

Loads the saved GNN weights (results/model_catgnn.pt); retrains the XGB
two-stage (fast). Same 2-week horizon / 1y window / masks as both parents.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, load_universe, gpu_lock, EXP
from eval_protocol import evaluate

HORIZON, TRAIN_THR, HID = 10, 0.10, 128

d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                        window_years=1)
print(f"dataset: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}",
      flush=True)
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
rel_all = d["rel"].to_numpy(dtype="float32")
dates_all = d.index.get_level_values("date")

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
y_ext_fit = y_ext_15[fit_m]
spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
with gpu_lock():
    stage1 = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0,
                           objective="binary:logistic", scale_pos_weight=spw,
                           eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], y_ext_fit,
               eval_set=[(d.loc[val_m, features], y_ext_15[val_m])],
               verbose=False)

def xgb_score(mask):
    p_ext = stage1.predict_proba(d.loc[mask, features])[:, 1]
    p_dir = stage2.predict_proba(d.loc[mask, features])[:, 1]
    return p_ext * (2.0 * p_dir - 1.0)

# ---------------------------------------------------------------- CatGNN (load)
import torch
import torch.nn as nn

med = d.loc[fit_m, features].median()
mu = d.loc[fit_m, features].mean()
sd = d.loc[fit_m, features].std().replace(0, 1.0)
Xall = ((d[features].fillna(med) - mu) / sd).clip(-5, 5).fillna(0.0) \
    .to_numpy(dtype="float32")
_, cat = load_universe()
cat_codes = pd.Series(pd.Categorical(
    d.index.get_level_values("ticker").map(cat)).codes,
    index=d.index).to_numpy()
NF = len(features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attnpool_cat(h, cid, attn_w):
    e = (h * attn_w).sum(-1)
    emax = torch.full((int(cid.max()) + 1,), -torch.inf, device=h.device)
    emax = emax.scatter_reduce(0, cid, e, reduce="amax")
    a = torch.exp(e - emax[cid])
    denom = torch.zeros_like(emax).scatter_add(0, cid, a)
    a = (a / denom[cid]).unsqueeze(-1)
    pooled = torch.zeros(int(cid.max()) + 1, h.shape[1], device=h.device)
    pooled = pooled.index_add(0, cid, h * a)
    return pooled[cid]


class CatGNN(nn.Module):
    def __init__(self, nf=NF, hid=HID):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, hid), nn.GELU())
        self.attn1 = nn.Parameter(torch.randn(hid) / hid ** 0.5)
        self.mp1 = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                 nn.Dropout(0.2))
        self.n1 = nn.LayerNorm(hid)
        self.attn2 = nn.Parameter(torch.randn(hid) / hid ** 0.5)
        self.mp2 = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                 nn.Dropout(0.2))
        self.n2 = nn.LayerNorm(hid)
        self.head = nn.Sequential(nn.Linear(hid + nf, hid), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(hid, 1))

    def forward(self, x, cid):
        z = self.inp(x)
        z = self.n1(z + self.mp1(torch.cat([z, attnpool_cat(z, cid, self.attn1)], -1)))
        z = self.n2(z + self.mp2(torch.cat([z, attnpool_cat(z, cid, self.attn2)], -1)))
        return self.head(torch.cat([z, x], -1)).squeeze(-1)


gnn = CatGNN().to(device)
gnn.load_state_dict(torch.load(EXP / "results" / "model_catgnn.pt",
                               map_location=device, weights_only=True))
gnn.eval()


def gnn_score(mask):
    idx = np.flatnonzero(mask)
    df = pd.DataFrame({"i": idx}, index=dates_all[idx])
    out = np.empty(len(idx), dtype="float32")
    pos = 0
    with torch.no_grad():
        for _, g in df.groupby(level=0):
            ix = g["i"].to_numpy()
            x = torch.from_numpy(Xall[ix]).to(device)
            cid = torch.from_numpy(cat_codes[ix].astype(np.int64)).to(device)
            out[pos:pos + len(ix)] = gnn(x, cid).cpu().numpy()
            pos += len(ix)
    return out

# ---------------------------------------------------------------- blend
def daily_z(mask, s):
    ser = pd.Series(s, index=d.index[mask])
    g = ser.groupby(level="date")
    return ((ser - g.transform("mean")) / g.transform("std").replace(0, 1.0)) \
        .to_numpy()


def daily_ic(mask, s):
    df = pd.DataFrame({"s": s, "rel": d.loc[mask, "rel"]}, index=d.index[mask])
    return float(df.groupby(level="date")
                 .apply(lambda g: g["s"].corr(g["rel"], method="spearman"))
                 .mean())


sx_val, sg_val = daily_z(val_m, xgb_score(val_m)), daily_z(val_m, gnn_score(val_m))
sx_te, sg_te = daily_z(te_m, xgb_score(te_m)), daily_z(te_m, gnn_score(te_m))

print("\nvalidation ICs: xgb={:.4f}  gnn={:.4f}".format(
    daily_ic(val_m, sx_val), daily_ic(val_m, sg_val)))
best_w, best_ic = None, -np.inf
for w in np.arange(0.0, 1.01, 0.1):
    ic = daily_ic(val_m, w * sx_val + (1 - w) * sg_val)
    print(f"  w_xgb={w:.1f}  val_ic={ic:+.4f}")
    if ic > best_ic:
        best_w, best_ic = w, ic
print(f"chosen blend weight (by val IC only): w_xgb={best_w:.1f}")

s_blend = best_w * sx_te + (1 - best_w) * sg_te
evaluate("blend_xgb_gnn", d, te_m, s_blend,
         extra={"w_xgb": round(float(best_w), 2), "horizon": HORIZON,
                "window_years": 1, "val_ic_at_w": round(best_ic, 4)})
