"""exp_10_combined — concat all 7 feature families and train the combined model."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, build_label, run_experiment,
                         save_features, EXP)

FAMILY_FILES = [
    "exp_01_voltail.parquet",
    "exp_02_peermom.parquet",
    "exp_03_peerext.parquet",
    "exp_04_liquidity.parquet",
    "exp_05_betaidio.parquet",
    "exp_06_catcontext.parquet",
    "exp_07_range.parquet",
]

def load_combined():
    try:
        frames = [pd.read_parquet(EXP / "features" / f) for f in FAMILY_FILES]
        X = pd.concat(frames, axis=1)
        X = X.loc[:, ~X.columns.duplicated(keep="first")]
        return X
    except MemoryError:
        print("MemoryError on concat -> falling back to incremental join")
        X = pd.read_parquet(EXP / "features" / FAMILY_FILES[0])
        for f in FAMILY_FILES[1:]:
            frame = pd.read_parquet(EXP / "features" / f)
            X = X.join(frame, how="outer", rsuffix="_dup")
            X = X.loc[:, [c for c in X.columns if not c.endswith("_dup")]]
        return X

X = load_combined()
print("combined design matrix:", X.shape, "features:", len(X.columns))

fields, cat = load_universe()
close = fields["Close"]
lbl = build_label(close, cat)
save_features("exp_10_combined", X)
res = run_experiment("exp_10_combined", X, lbl, close.index,
                     params={"iterations": 3000})
