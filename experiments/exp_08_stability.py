"""
exp_08_stability — stability audit of the volatility tail effect.

A prior analysis on the 2026 holdout found the vol signal is a TAIL effect:
mean next-5d peer-relative return is flat across vol deciles 1-9 but jumps
in decile 10. This script tests whether that holds across years.

PART A: per-calendar-year decile tables (pure statistics, no model) for
        vol_20d and vol_60d_rel, 2019..2026.
PART B: walk-forward CatBoost on a small vol feature set, test years
        2021..2026, expanding purged train windows via holdout_masks.
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from exp_harness import (load_universe, stack, assemble, build_label,
                         loo_peer_mean, cat_pct_rank, holdout_masks,
                         gpu_lock, DEFAULT_PARAMS, EXP)

RESULTS = EXP / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("loading universe ...", flush=True)
fields, cat = load_universe()
close = fields["Close"]

ret = close.pct_change(fill_method=None)
vol_20d = ret.rolling(20).std()
vol_60d = ret.rolling(60).std()
vol_60d_rel = vol_60d - loo_peer_mean(vol_60d, cat)

print("building label ...", flush=True)
lbl = build_label(close, cat)

# ---------------------------------------------------------------------------
# PART A — per-year decile tables (no model)
# ---------------------------------------------------------------------------
print("\n=== PART A: decile-table stability ===", flush=True)

featsA = {"vol_20d": stack(vol_20d), "vol_60d_rel": stack(vol_60d_rel)}
dfA = pd.DataFrame(featsA).join(lbl[["rel"]], how="inner")
years = range(2019, 2027)

rows = []
for yr in years:
    yr_mask = dfA.index.get_level_values("date").year == yr
    row = {"year": yr}
    for feat in featsA:
        sub = dfA.loc[yr_mask, [feat, "rel"]].dropna()
        if len(sub) < 100:
            row.update({f"{feat}_d1_bps": np.nan, f"{feat}_mid_bps": np.nan,
                        f"{feat}_d10_bps": np.nan, f"{feat}_d10_t": np.nan,
                        f"{feat}_n": len(sub)})
            continue
        dec = pd.qcut(sub[feat].rank(method="first"), 10, labels=False) + 1
        rel_bps = sub["rel"] * 1e4
        d1 = rel_bps[dec == 1]
        mid = rel_bps[(dec >= 2) & (dec <= 9)]
        d10 = rel_bps[dec == 10]
        t10 = d10.mean() / (d10.std(ddof=1) / np.sqrt(len(d10)))
        row.update({f"{feat}_d1_bps": round(d1.mean(), 2),
                    f"{feat}_mid_bps": round(mid.mean(), 2),
                    f"{feat}_d10_bps": round(d10.mean(), 2),
                    f"{feat}_d10_t": round(t10, 2),
                    f"{feat}_n": len(sub)})
    rows.append(row)

decile_tbl = pd.DataFrame(rows).set_index("year")
pd.set_option("display.width", 200)
print(decile_tbl.to_string())
decile_tbl.to_csv(RESULTS / "exp_08_stability_deciles.csv")
print("saved ->", RESULTS / "exp_08_stability_deciles.csv", flush=True)

# ---------------------------------------------------------------------------
# PART B — walk-forward model stability
# ---------------------------------------------------------------------------
print("\n=== PART B: walk-forward model stability ===", flush=True)
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

hl_range_20d = ((fields["High"] - fields["Low"]) / close).rolling(20).mean()
vol_20d_rel = vol_20d - loo_peer_mean(vol_20d, cat)
vol_20d_catrank = cat_pct_rank(vol_20d, cat)

X = assemble({
    "vol_20d": vol_20d,
    "vol_60d": vol_60d,
    "hl_range_20d": hl_range_20d,
    "vol_20d_rel": vol_20d_rel,
    "vol_60d_rel": vol_60d_rel,
    "vol_20d_catrank": vol_20d_catrank,
})
features = list(X.columns)

d = X.join(lbl, how="inner").dropna(subset=["target"])
d = d.dropna(subset=features, how="all")
d["target"] = d["target"].astype("int8")
date_index = d.index.get_level_values("date")

params = {**DEFAULT_PARAMS, "iterations": 1000, "verbose": 0}

year_rows = []
for yr in (2021, 2022, 2023, 2024, 2025, 2026):
    fit_mask, val_mask, te_mask = holdout_masks(date_index, close.index,
                                                test_year=yr)
    n_fit, n_val, n_te = int(fit_mask.sum()), int(val_mask.sum()), int(te_mask.sum())
    print(f"\n[{yr}] n_fit={n_fit} n_val={n_val} n_test={n_te}", flush=True)
    if n_fit == 0 or n_val == 0 or n_te == 0:
        print(f"[{yr}] skipped (empty split)", flush=True)
        year_rows.append({"test_year": yr, "n_train": n_fit, "n_test": n_te,
                          "macro_f1": np.nan, "accuracy": np.nan,
                          "rank_ic": np.nan, "best_iteration": np.nan})
        continue

    with gpu_lock():
        model = CatBoostClassifier(early_stopping_rounds=100,
                                   use_best_model=True, **params)
        model.fit(d.loc[fit_mask, features], d.loc[fit_mask, "target"],
                  eval_set=(d.loc[val_mask, features],
                            d.loc[val_mask, "target"]))

    y_te = d.loc[te_mask, "target"]
    pred = np.asarray(model.predict(d.loc[te_mask, features])).ravel().astype(int)
    proba = model.predict_proba(d.loc[te_mask, features])[:, 2]
    ic = (pd.DataFrame({"s": proba, "rel": d.loc[te_mask, "rel"]},
                       index=y_te.index)
          .groupby(level="date")
          .apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean())

    row = {"test_year": yr, "n_train": n_fit, "n_test": n_te,
           "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
           "accuracy": round(float(accuracy_score(y_te, pred)), 4),
           "rank_ic": round(float(ic), 4),
           "best_iteration": int(model.get_best_iteration()
                                 or params["iterations"])}
    print(f"[{yr}] F1={row['macro_f1']} acc={row['accuracy']} "
          f"IC={row['rank_ic']} best_iter={row['best_iteration']}", flush=True)
    year_rows.append(row)

year_tbl = pd.DataFrame(year_rows).set_index("test_year")
print("\nper-year model metrics:")
print(year_tbl.to_string())
year_tbl.to_csv(RESULTS / "exp_08_stability_years.csv")
print("saved ->", RESULTS / "exp_08_stability_years.csv", flush=True)

# ---------------------------------------------------------------------------
# summary json
# ---------------------------------------------------------------------------
summary = {
    "name": "exp_08_stability",
    "per_year_model": year_tbl.reset_index().replace({np.nan: None})
                              .to_dict(orient="records"),
    "decile10_bps_per_year": {
        str(yr): {
            "vol_20d_d10_bps": (None if pd.isna(decile_tbl.loc[yr, "vol_20d_d10_bps"])
                                else float(decile_tbl.loc[yr, "vol_20d_d10_bps"])),
            "vol_20d_d10_t": (None if pd.isna(decile_tbl.loc[yr, "vol_20d_d10_t"])
                              else float(decile_tbl.loc[yr, "vol_20d_d10_t"])),
            "vol_60d_rel_d10_bps": (None if pd.isna(decile_tbl.loc[yr, "vol_60d_rel_d10_bps"])
                                    else float(decile_tbl.loc[yr, "vol_60d_rel_d10_bps"])),
            "vol_60d_rel_d10_t": (None if pd.isna(decile_tbl.loc[yr, "vol_60d_rel_d10_t"])
                                  else float(decile_tbl.loc[yr, "vol_60d_rel_d10_t"])),
        } for yr in decile_tbl.index
    },
    "features": features,
}
(RESULTS / "exp_08_stability.json").write_text(json.dumps(summary, indent=2))
print("saved ->", RESULTS / "exp_08_stability.json")
print("\nDONE", flush=True)
