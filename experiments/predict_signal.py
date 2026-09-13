"""predict_signal — emit today's ranked over/under-perform calls.

Loads the artifacts saved by train_production.py (XGB two-stage + CatGNN +
blend weight), scores every fund on the latest available date (or --date
YYYY-MM-DD), and writes the gated call list:

    python experiments/predict_signal.py [--date 2026-07-10] [--k 0.03]

Output: experiments/production/signal_<date>.csv with every fund's blend
score, plus the top-K% conviction calls flagged with direction. Prints the
top/bottom 15 calls. Signal horizon: forward 10 trading days (2 weeks),
peer-relative within category.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from exp_harness import load_universe, EXP, SWEEP_FAMILIES
import nn_common as nc

PROD = EXP / "production"

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest)")
ap.add_argument("--k", type=float, default=0.03, help="conviction gate")
args = ap.parse_args()

config = json.loads((PROD / "config.json").read_text())
features = config["features"]
cat_map = {c: i for i, c in enumerate(config["categories"])}

# ---------------------------------------------------------------- features
frames = [pd.read_parquet(EXP / "features" / f) for f in SWEEP_FAMILIES
          if (EXP / "features" / f).exists()]
X = pd.concat(frames, axis=1)
X = X.loc[:, ~X.columns.duplicated(keep="first")].astype("float32")
X = X.mask(np.isinf(X))[features]

dates = X.index.get_level_values("date")
target_date = pd.Timestamp(args.date) if args.date else dates.max()
day = X[dates == target_date].dropna(how="all")
if day.empty:
    sys.exit(f"no feature rows for {target_date.date()}")
print(f"scoring {len(day)} funds for {target_date.date()} "
      f"(model trained {config['trained']})")

_, cat = load_universe()
day_cat = day.index.get_level_values("ticker").map(cat)
cat_codes = np.array([cat_map.get(c, len(cat_map)) for c in day_cat])

# ---------------------------------------------------------------- XGB score
import xgboost as xgb
b1, b2 = xgb.Booster(), xgb.Booster()
b1.load_model(str(PROD / "stage1.ubj"))
b2.load_model(str(PROD / "stage2.ubj"))
dm = xgb.DMatrix(day, feature_names=features)
p_ext, p_dir = b1.predict(dm), b2.predict(dm)
s_xgb = p_ext * (2.0 * p_dir - 1.0)

# ---------------------------------------------------------------- GNN score
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = pd.read_parquet(PROD / "scaler.parquet")
Xn = nc.standardize(day, scaler["median"], scaler["mean"], scaler["std"])
gnn = nc.CatGNN(len(features), config["gnn_hidden"]).to(device)
gnn.load_state_dict(torch.load(PROD / "catgnn.pt", map_location=device,
                               weights_only=True))
gnn.eval()
with torch.no_grad():
    s_gnn = gnn(torch.from_numpy(Xn).to(device),
                torch.from_numpy(cat_codes.astype(np.int64)).to(device)) \
        .cpu().numpy()

# ---------------------------------------------------------------- blend + gate
def z(s):
    s = np.asarray(s, dtype=float)
    return (s - s.mean()) / (s.std() or 1.0)

w = config["w_xgb"]
score = w * z(s_xgb) + (1 - w) * z(s_gnn)

out = pd.DataFrame({
    "ticker": day.index.get_level_values("ticker"),
    "category": day_cat,
    "score": score,
    "p_extreme": p_ext,
    "p_direction_up": p_dir,
}).sort_values("score", ascending=False).reset_index(drop=True)
n_gate = max(int(len(out) * args.k), 1)
med_s = out["score"].median()
out["conviction"] = (out["score"] - med_s).abs()
gated = out.nlargest(n_gate, "conviction").copy()
gated["call"] = np.where(gated["score"] >= med_s, "OVERPERFORM", "UNDERPERFORM")
out["call"] = ""
out.loc[gated.index, "call"] = gated["call"]

path = PROD / f"signal_{target_date.date()}.csv"
out.to_csv(path, index=False)
print(f"\nsaved -> {path}")
print(f"\ngated calls (top {args.k:.0%} conviction, n={n_gate}, "
      f"horizon = next 10 trading days, peer-relative):")
cols = ["ticker", "category", "score", "call"]
print("\n-- strongest OVERPERFORM calls --")
print(gated[gated["call"] == "OVERPERFORM"].head(15)[cols]
      .to_string(index=False))
print("\n-- strongest UNDERPERFORM calls --")
print(gated[gated["call"] == "UNDERPERFORM"]
      .sort_values("score").head(15)[cols].to_string(index=False))
