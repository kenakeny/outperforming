"""
train_windows.py — train every (horizon x training-window) model and score them honestly.

  horizons : 5-day (skip 0) and 20-day (skip 5)
  windows  : trailing 6M / 1Y / 2Y / 3Y / 4Y / 5Y before the 2026 test year
  features : features_v3 (44, identical across every model)
  metrics  : rank IC (mean, t-stat, positive-day %) over ALL 2026 dates, leakage-free,
             plus a non-overlapping top/bottom-quintile long-short backtest of $100k.

Writes window_results.json and window_importances.json to the scratchpad.
"""
import pathlib, yaml, json
import numpy as np, pandas as pd
from catboost import CatBoostClassifier

ROOT = pathlib.Path(__file__).resolve().parent
cfg  = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RAW, PROC = ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["processed"]
OUT = pathlib.Path(r"C:\Windows\TEMP\claude\C--Users-user-Desktop-CS-overp-outperforming\6aa5ec25-518d-42ab-881a-03f7fdfd17e2\scratchpad")

MIN_GROUP = 3
TEST_YEAR = 2026
CAPITAL   = 100_000
QUINT     = 0.20
MIN_PRICE = 5.0
WINDOWS   = {"6M": 6, "1Y": 12, "2Y": 24, "3Y": 36, "4Y": 48, "5Y": 60}
HORIZONS  = {"5d": (5, 0), "20d": (20, 5)}

print("loading features + prices ...", flush=True)
feats = pd.read_parquet(PROC / "features_v3.parquet")
feats.index = feats.index.set_levels(pd.to_datetime(feats.index.levels[0]), level="date")
FEATURES = list(feats.columns)

metadata = pd.read_parquet(RAW / "metadata.parquet")
prices = pd.read_parquet(RAW / "prices.parquet")
prices.index.name = "date"; prices.columns.name = "ticker"
cat = metadata["category"].reindex(prices.columns)
counts = cat.value_counts()
keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
cols = cat[cat.isin(keep)].index
px, cat = prices[cols], cat[cols]
price_l = prices.stack(); price_l.index.names = ["date", "ticker"]


def _stack(df):
    s = df.stack(future_stack=True); s.index.names = ["date", "ticker"]; return s


def build_dataset(H, SKIP):
    fwd = px.shift(-(H + SKIP)) / px.shift(-SKIP) - 1.0
    gs = fwd.T.groupby(cat).transform("sum").T
    gn = fwd.notna().T.groupby(cat).transform("sum").T
    peer = (gs - fwd.fillna(0)) / (gn - fwd.notna().astype(int)).replace(0, np.nan)
    rel = fwd - peer
    fwd_l, rel_l = _stack(fwd), _stack(rel)
    target = (rel_l.dropna().groupby(level="date")
              .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=[0, 1, 2])).astype("int8"))
    d = feats.copy()
    d["target"] = target.reindex(d.index)
    d["fwd_ret"] = fwd_l.reindex(d.index)
    d = d.dropna(subset=["target"]); d["target"] = d["target"].astype("int8")
    return d


def rank_ic(score, truth, idx):
    df = pd.DataFrame({"s": score, "t": truth.values}, index=idx)
    daily = df.groupby(level="date").apply(lambda x: x["s"].corr(x["t"], method="spearman")).dropna()
    ir = daily.mean() / daily.std()
    return dict(mean=float(daily.mean()), t=float(ir * np.sqrt(len(daily))),
               pos=float((daily > 0).mean()), n=int(len(daily)))


def long_short(score, fwd, price, idx, HOLD):
    bt = pd.DataFrame({"score": score, "fwd": fwd.values, "px": price.values}, index=idx).dropna()
    cap_mv = 0.5 if HOLD == 20 else 0.25
    bt = bt[(bt["px"] >= MIN_PRICE) & (bt["fwd"].abs() <= cap_mv)]
    dates = np.sort(bt.index.get_level_values("date").unique())
    cap = CAPITAL; rets = []
    for d in dates[::HOLD]:
        day = bt.xs(d, level="date")
        n = max(1, int(round(len(day) * QUINT)))
        top = day.nlargest(n, "score")["fwd"].mean()
        bot = day.nsmallest(n, "score")["fwd"].mean()
        s = top - bot
        cap *= (1 + s); rets.append(s)
    rets = np.array(rets)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252 / HOLD)) if rets.std() > 0 else 0.0
    return dict(final=float(cap), total=float(cap / CAPITAL - 1),
                avg=float(rets.mean()), win=float((rets > 0).mean()),
                sharpe=sharpe, periods=int(len(rets)))


results = []
importances = {}
params = dict(iterations=2000, learning_rate=0.04, depth=7, l2_leaf_reg=3,
              loss_function="MultiClass", eval_metric="TotalF1:average=Macro",
              random_seed=42, task_type="GPU", verbose=False)

for hname, (H, SKIP) in HORIZONS.items():
    print(f"\n=== horizon {hname} : building label ===", flush=True)
    d = build_dataset(H, SKIP)
    EMB = H + SKIP
    di = d.index.get_level_values("date")
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
    cut = test_start - pd.offsets.BDay(EMB)
    te = di >= test_start
    Xte, fwd_te = d.loc[te, FEATURES], d.loc[te, "fwd_ret"]
    px_te = price_l.reindex(d.loc[te].index)
    imp_acc = np.zeros(len(FEATURES))

    for wname, wm in WINDOWS.items():
        tstart = test_start - pd.DateOffset(months=wm)
        tr = (di >= tstart) & (di <= cut)
        tr_dates = np.sort(di[tr].unique())
        vstart = tr_dates[int(len(tr_dates) * 0.90)]
        vcut = vstart - pd.offsets.BDay(EMB)
        fit = tr & (di <= vcut); val = tr & (di >= vstart)

        m = CatBoostClassifier(early_stopping_rounds=100, use_best_model=True, **params)
        m.fit(d.loc[fit, FEATURES], d.loc[fit, "target"],
              eval_set=(d.loc[val, FEATURES], d.loc[val, "target"]))
        proba = m.predict_proba(Xte)[:, 2]
        ic = rank_ic(proba, fwd_te, Xte.index)
        ls = long_short(proba, fwd_te, px_te, Xte.index, H)
        imp_acc += np.array(m.get_feature_importance())
        row = dict(horizon=hname, window=wname, fit_rows=int(fit.sum()),
                   ic=round(ic["mean"], 4), ic_t=round(ic["t"], 2), ic_pos=round(ic["pos"], 3),
                   ls_final=round(ls["final"]), ls_total=round(ls["total"], 4),
                   ls_sharpe=round(ls["sharpe"], 2), ls_win=round(ls["win"], 2), ls_periods=ls["periods"])
        results.append(row)
        print(f"  {hname:>3} {wname:>3} | fit={row['fit_rows']:>8,} | IC={row['ic']:+.3f} "
              f"t={row['ic_t']:+.2f} | L/S ${row['ls_final']:>8,} ({row['ls_total']:+.1%}) "
              f"Sharpe={row['ls_sharpe']:+.2f}", flush=True)

    importances[hname] = dict(zip(FEATURES, (imp_acc / len(WINDOWS)).round(3).tolist()))

json.dump(results, open(OUT / "window_results.json", "w"), indent=2)
json.dump(importances, open(OUT / "window_importances.json", "w"), indent=2)
print("\nSAVED window_results.json + window_importances.json", flush=True)
