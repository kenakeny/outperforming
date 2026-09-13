"""exp_13_composition — ETF composition (holdings) feature family.

Static per-fund descriptors from yfinance funds_data (fetched by
fetch_composition.py): sector weights, asset-class mix, top-10 holdings
concentration, valuation yields — plus peer-relative versions (distance of
the fund's sector mix from its category's average mix, concentration rank
within category, etc.), since "is this fund composed differently from its
peers" is the natural composition analogue of the peer-relative returns that
drive the label.

CAVEAT (known, accepted): composition is a TODAY-snapshot — yfinance has no
history — so these are static descriptors like `category` itself, constant
across all dates. They trivially pass trailing-computation checks but carry
as-of-today bias for old sample dates; judge them mainly on the 2026 holdout,
which is closest in time to the snapshot.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, assemble, build_label, save_features,
                         run_experiment, cat_pct_rank, RAW)

fields, cat = load_universe()
close = fields["Close"]
tickers = close.columns

comp = pd.read_parquet(RAW / "etf_composition.parquet")
comp = comp[comp["ok"] == 1].set_index("ticker").reindex(tickers)
print(f"composition coverage: {comp['ok'].notna().sum()}/{len(tickers)} tickers")

SEC_COLS = [c for c in comp.columns if c.startswith("sec_")]
# funds with no sector data but real asset-class data (bond funds) -> sectors 0
has_ac = comp[["pos_stock", "pos_bond", "pos_cash"]].notna().any(axis=1)
comp.loc[has_ac, SEC_COLS] = comp.loc[has_ac, SEC_COLS].fillna(0.0)

# ------------------------------------------------- peer-relative derivations
cat_ser = cat.reindex(tickers)

# sector-mix distance from category average mix (L1 / 2 -> 0..1)
sec = comp[SEC_COLS]
cat_mix = sec.groupby(cat_ser).transform("mean")
sec_dist_cat = (sec - cat_mix).abs().sum(axis=1, min_count=1) / 2.0

# asset-class tilt vs category
stock_pos_catrel = comp["pos_stock"] - comp["pos_stock"].groupby(cat_ser).transform("mean")

# concentration + valuation, absolute and later ranked within category
static = {
    **{c: comp[c] for c in SEC_COLS},
    "pos_stock": comp["pos_stock"],
    "pos_bond": comp["pos_bond"],
    "pos_cash": comp["pos_cash"],
    "top10_conc": comp["top10_conc"],
    "top1_weight": comp["top1_weight"],
    "earn_yield": comp["earn_yield"],
    "book_yield": comp["book_yield"],
    "sec_dist_cat": sec_dist_cat,
    "stock_pos_catrel": stock_pos_catrel,
}

# broadcast static per-ticker values to wide (date x ticker), masked to days
# the fund actually has a price
mask = close.notna()
feat_dict = {}
for name, s in static.items():
    wide = pd.DataFrame(np.broadcast_to(s.to_numpy(dtype=float)[None, :],
                                        close.shape).copy(),
                        index=close.index, columns=close.columns)
    feat_dict[name] = wide.where(mask)

# within-category percentile ranks (constant over time but peer-relative)
feat_dict["top10_conc_catrank"] = cat_pct_rank(feat_dict["top10_conc"], cat)
feat_dict["earn_yield_catrank"] = cat_pct_rank(feat_dict["earn_yield"], cat)

X = assemble(feat_dict)
lbl = build_label(close, cat)
save_features("exp_13_composition", X)
run_experiment("exp_13_composition", X, lbl, close.index)
