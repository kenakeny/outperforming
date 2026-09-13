"""opup_f1_frontier — maximize F1 of OVERPERFORM and UNDERPERFORM directly.

The user's target metric is the F1 of the two directional classes (neutral
is irrelevant). Two levers move those F1s, neither of which needs new
training:

  1. class definition: how much of the daily cross-section counts as
     over/underperform (tercile 33/33/33 down to 90/5/5) -- wider classes
     mechanically support higher F1;
  2. call volume q: call "over" for the top q% of the score per day and
     "under" for the bottom q% -- matched-rate calling (q = class rate) is
     NOT the F1 optimum; q is tuned on VALIDATION ONLY to maximize
     (F1_over + F1_under)/2, then applied once to the 2026 test.

Signal: the saved production blend (XGB two-stage x CatGNN, w from config),
scored on the real feature data (experiments/features/*.parquet built from
data/raw/market_data.parquet). For every (definition, tuned q): test F1_over,
F1_under, and the money (net per call at 20bps round trip, annualized
long-short return, Sharpe) so the F1/money trade-off is explicit.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from exp_harness import load_sweep_dataset, load_universe, EXP
import nn_common as nc

PROD = EXP / "production"
HORIZON, COST_RT = 10, 0.002
TAILS = [1 / 3, 0.15, 0.10, 0.05]
Q_GRID = np.arange(0.03, 0.51, 0.01)

config = json.loads((PROD / "config.json").read_text())
features, w = config["features"], config["w_xgb"]

d, _, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                 window_years=1)
day_pct = d.groupby(level="date")["rel"].rank(pct=True)
dates_all = d.index.get_level_values("date")
print(f"val={val_m.sum()} test={te_m.sum()}", flush=True)

# ---------------------------------------------------------------- blend score
import xgboost as xgb
import torch

b1, b2 = xgb.Booster(), xgb.Booster()
b1.load_model(str(PROD / "stage1.ubj"))
b2.load_model(str(PROD / "stage2.ubj"))

scaler = pd.read_parquet(PROD / "scaler.parquet")
cat_map = {c: i for i, c in enumerate(config["categories"])}
_, cat = load_universe()
cat_codes = np.array([cat_map.get(c, len(cat_map)) for c in
                      d.index.get_level_values("ticker").map(cat)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn = nc.CatGNN(len(features), config["gnn_hidden"]).to(device)
gnn.load_state_dict(torch.load(PROD / "catgnn.pt", map_location=device,
                               weights_only=True))
gnn.eval()
Xn_all = nc.standardize(d[features], scaler["median"], scaler["mean"],
                        scaler["std"])


def blend_score(mask):
    dm = xgb.DMatrix(d.loc[mask, features], feature_names=features)
    s_x = b1.predict(dm) * (2.0 * b2.predict(dm) - 1.0)
    days = nc.day_slices(mask, dates_all, cat_codes)
    s_g = nc.day_scores(gnn, days, Xn_all, device)
    ser_x = pd.Series(s_x, index=d.index[mask])
    ser_g = pd.Series(s_g, index=d.index[mask])
    def z(ser):
        g = ser.groupby(level="date")
        return (ser - g.transform("mean")) / g.transform("std").replace(0, 1)
    return (w * z(ser_x) + (1 - w) * z(ser_g)).to_numpy()


s_val, s_te = blend_score(val_m), blend_score(te_m)
val = pd.DataFrame({"s": s_val, "pct": day_pct[val_m],
                    "rel": d.loc[val_m, "rel"]}, index=d.index[val_m])
te = pd.DataFrame({"s": s_te, "pct": day_pct[te_m],
                   "rel": d.loc[te_m, "rel"]}, index=d.index[te_m])
for df in (val, te):
    df["s_pct"] = df.groupby(level="date")["s"].rank(pct=True)


def opup_f1(df, tail, q):
    y = np.select([df["pct"] <= tail, df["pct"] >= 1 - tail], [0, 2], default=1)
    p = np.select([df["s_pct"] <= q, df["s_pct"] >= 1 - q], [0, 2], default=1)
    f1_u, f1_o = f1_score(y, p, average=None, labels=[0, 2], zero_division=0)
    return float(f1_u), float(f1_o), y, p


rows = []
for tail in TAILS:
    best_q, best = None, -1
    for q in Q_GRID:
        f1_u, f1_o, _, _ = opup_f1(val, tail, q)
        if (f1_u + f1_o) / 2 > best:
            best, best_q = (f1_u + f1_o) / 2, float(q)
    f1_u, f1_o, y_te, p_te = opup_f1(te, tail, best_q)

    calls = te[p_te != 1].copy()
    sign = np.where(p_te[p_te != 1] == 2, 1.0, -1.0)
    calls["pnl"] = sign * calls["rel"] - COST_RT
    day_pnl = calls.groupby(level="date")["pnl"].mean()
    ann = np.sqrt(252 / HORIZON)
    split = ("33/33/33" if tail > 0.3 else
             f"{int((1 - 2 * tail) * 100)}/{int(tail * 100)}/{int(tail * 100)}")
    rows.append({
        "split": split, "tuned_q": round(best_q, 2),
        "calls_per_day": round(float(calls.groupby(level='date').size().mean()), 0),
        "f1_overperform": round(f1_o, 4), "f1_underperform": round(f1_u, 4),
        "net_per_call_10d": round(float(calls["pnl"].mean()), 5),
        "ann_ret_net": round(float(day_pnl.mean() * 252 / HORIZON), 4),
        "sharpe_net": round(float(day_pnl.mean() / day_pnl.std() * ann), 3),
    })
    print(rows[-1], flush=True)

res = pd.DataFrame(rows)
print("\n===== op/up F1 frontier (production blend, q tuned on val only) =====")
print(res.to_string(index=False))
res.to_csv(EXP / "results" / "opup_f1_frontier.csv", index=False)
print("\nsaved -> experiments/results/opup_f1_frontier.csv")
