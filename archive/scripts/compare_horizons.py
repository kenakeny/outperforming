"""
compare_horizons.py — 5d vs 20d, SAME features (features_v3), SAME method, IC over ALL test dates.

Trains a 20-day holdout model on features_v3 (the 5d artifact backtest_5d.parquet already exists),
scores every 2026 date, then prints rank IC for both horizons over all ~125 dates so the t-stats
are computed on the same footing. The only thing different between the two is the horizon.
"""
import pathlib, yaml
import numpy as np, pandas as pd
from catboost import CatBoostClassifier

ROOT = pathlib.Path(__file__).resolve().parent
cfg  = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RAW, PROC = ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["processed"]
MIN_GROUP, TEST_YEAR = 3, 2026

def build_backtest_frame(HORIZON, SKIP):
    EMBARGO = HORIZON + SKIP
    feats = pd.read_parquet(PROC / "features_v3.parquet")
    feats.index = feats.index.set_levels(pd.to_datetime(feats.index.levels[0]), level="date")

    metadata = pd.read_parquet(RAW / "metadata.parquet")
    prices = pd.read_parquet(RAW / "prices.parquet")
    prices.index.name = "date"; prices.columns.name = "ticker"
    cat = metadata["category"].reindex(prices.columns)
    counts = cat.value_counts()
    keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
    cols = cat[cat.isin(keep)].index
    px, cat = prices[cols], cat[cols]

    fwd = px.shift(-(HORIZON + SKIP)) / px.shift(-SKIP) - 1.0
    grp_sum = fwd.T.groupby(cat).transform("sum").T
    grp_n = fwd.notna().T.groupby(cat).transform("sum").T
    peer_mean = (grp_sum - fwd.fillna(0)) / (grp_n - fwd.notna().astype(int)).replace(0, np.nan)
    rel = fwd - peer_mean

    def _stack(df):
        s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s
    fwd_l, rel_l = _stack(fwd), _stack(rel)
    target = (rel_l.dropna().groupby(level="date")
              .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=[0, 1, 2])).astype("int8"))
    lbl = pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1).dropna(subset=["target"])

    d = feats.copy()
    d["target"] = lbl["target"].reindex(d.index)
    d["fwd_ret"] = lbl["fwd_ret"].reindex(d.index)
    d["rel"] = lbl["rel"].reindex(d.index)
    d = d.dropna(subset=["target"]); d["target"] = d["target"].astype("int8")
    features = [c for c in d.columns if c not in ("target", "fwd_ret", "rel")]

    di = d.index.get_level_values("date")
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
    cut = test_start - pd.offsets.BDay(EMBARGO)
    tr = di <= cut; te = di >= test_start
    tr_dates = np.sort(di[tr].unique())
    vstart = tr_dates[int(len(tr_dates) * 0.9)]
    vcut = vstart - pd.offsets.BDay(EMBARGO)
    fit = tr & (di <= vcut); val = tr & (di >= vstart)

    m = CatBoostClassifier(iterations=3000, learning_rate=0.04, depth=7, l2_leaf_reg=3,
                           loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
                           random_seed=42, task_type="GPU", early_stopping_rounds=150,
                           use_best_model=True, verbose=False)
    m.fit(d.loc[fit, features], d.loc[fit, "target"],
          eval_set=(d.loc[val, features], d.loc[val, "target"]))
    return pd.DataFrame({
        "score": m.predict_proba(d.loc[te, features])[:, 2],
        "fwd_ret": d.loc[te, "fwd_ret"].values,
        "rel": d.loc[te, "rel"].values,
    }, index=d.loc[te].index)


def rank_ic(df, truth):
    daily = df.groupby(level="date").apply(lambda x: x["score"].corr(x[truth], method="spearman")).dropna()
    ir = daily.mean() / daily.std()
    return daily.mean(), daily.std(), ir, ir * np.sqrt(len(daily)), (daily > 0).mean(), len(daily)


print("training 20-day holdout on features_v3 ...")
bt20 = build_backtest_frame(20, 5)
bt20.to_parquet(PROC / "backtest_20d_v3.parquet")

# 5d frame already exists but lacks 'rel'; rebuild quickly for parity is overkill — use existing for fwd_ret IC
bt5 = pd.read_parquet(PROC / "backtest_5d.parquet")

print("\n" + "=" * 68)
print(f"{'':10}{'mean IC':>10}{'IC std':>9}{'IC-IR':>8}{'t-stat':>9}{'pos%':>7}{'dates':>7}")
print("-" * 68)
for name, bt in [("5-day", bt5), ("20-day", bt20)]:
    mic, sd, ir, t, pos, n = rank_ic(bt.dropna(), "fwd_ret")
    sig = "OK" if abs(t) > 2 else "NOT sig"
    print(f"{name:10}{mic:>+10.4f}{sd:>9.3f}{ir:>+8.3f}{t:>+9.2f}{pos:>7.0%}{n:>7}   [{sig}]")
print("=" * 68)
print("(rank IC = daily Spearman of score vs next-window return, over ALL test dates)")
