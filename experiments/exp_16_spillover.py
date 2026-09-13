"""exp_16_spillover — peer lead-lag / category rotation features.

"My closest peers moved; I haven't yet." The GNN showed same-day peer state
carries signal; these features give the trees the LAGGED version (spillover
with timing), plus cross-category rotation context. All strictly trailing.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, loo_peer_mean)

fields, cat = load_universe()
close = fields["Close"]
r1 = close.pct_change(fill_method=None)
ret5 = close.pct_change(5, fill_method=None)
ret20 = close.pct_change(20, fill_method=None)

peer5 = loo_peer_mean(ret5, cat)
peer20 = loo_peer_mean(ret20, cat)

# 1-2. spillover gap: peers' trailing move minus own, as of yesterday
spill_gap_5d = (peer5 - ret5).shift(1)
spill_gap_20d = (peer20 - ret20).shift(1)

# 3. does the fund follow its category? corr(own today, category yesterday)
cat_r1 = loo_peer_mean(r1, cat)
follower_60d = r1.rolling(60).corr(cat_r1.shift(1))

# 4. follower x spillover: laggards that historically catch up
follow_x_spill = follower_60d * spill_gap_5d

# 5-6. cross-category rotation: rank of the fund's category's return among
# all categories (is my whole category hot or cold right now?)
cats = pd.Series(cat)
cat_ret20_by_cat = ret20.T.groupby(cats).mean().T          # date x category
cat_rank20 = cat_ret20_by_cat.rank(axis=1, pct=True)
cat_ret5_by_cat = ret5.T.groupby(cats).mean().T
cat_rank5 = cat_ret5_by_cat.rank(axis=1, pct=True)

def to_fund(frame_by_cat):
    m = frame_by_cat.reindex(columns=cats.unique())
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    vals = m[cats.reindex(close.columns)].to_numpy()
    out.iloc[:, :] = vals
    return out.where(close.notna())

cat_mom_rank_20d = to_fund(cat_rank20)
cat_mom_rank_5d = to_fund(cat_rank5)

# 7. category rotation persistence: 60d autocorr of the category's daily
# relative return (do category winners repeat lately?)
cat_rel_r1 = cat_r1  # LOO category mean daily return per fund's category
cat_persist_60d = cat_rel_r1.rolling(60).corr(cat_rel_r1.shift(1))

feat_dict = {
    "spill_gap_5d": spill_gap_5d,
    "spill_gap_20d": spill_gap_20d,
    "follower_60d": follower_60d,
    "follow_x_spill": follow_x_spill,
    "cat_mom_rank_20d": cat_mom_rank_20d,
    "cat_mom_rank_5d": cat_mom_rank_5d,
    "cat_persist_60d": cat_persist_60d,
}

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_16_spillover", X)
run_experiment("exp_16_spillover", X, lbl, close.index)
