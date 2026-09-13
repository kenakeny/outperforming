"""
train_universe.py -- train and SAVE a model on the universe-wide label.

Everything in models/ comes from train_models.py, which uses dataset_v3's
WITHIN-CATEGORY label. Those scores only mean something against a fund's own
peers -- ranking the whole universe by them builds a portfolio of "best bond
fund + best tech fund", which underperforms the universe it was drawn from.

This trains on core.panel's universe-wide label instead, so the scores are
comparable across every fund and can drive a cross-sectional portfolio.

    python train_universe.py                       # xgboost, h=20, test 2026
    python train_universe.py --kind catboost --horizon 5
"""
import argparse
import pathlib

import core

MODELS = pathlib.Path(__file__).resolve().parent / "models"
MODELS.mkdir(exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="xgboost",
                   choices=["xgboost", "catboost", "lightgbm", "logreg"])
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--test-year", type=int, default=2026)
    p.add_argument("--min-adv", type=float, default=50e6)
    a = p.parse_args()

    d, tidx = core.panel(a.horizon, min_adv=a.min_adv)
    cols = [c for c in d.columns if c not in core.LABEL_COLS]
    dd = d.reset_index()

    # train on everything up to `horizon` trading days before the test year --
    # the purge must equal the label horizon or the training data overlaps test
    import pandas as pd

    import common
    test_start = pd.Timestamp(f"{a.test_year}-01-01")
    cut = common.purge_cutoff(tidx, test_start, a.horizon)
    common.assert_no_overlap(tidx, cut, test_start, a.horizon)
    tr = dd[dd["date"] <= cut]
    print(f"panel {len(dd):,} rows | {len(cols)} features | "
          f"train {len(tr):,} rows through {cut.date()} "
          f"(purged {a.horizon}d before {a.test_year})")

    m = core.fit(a.kind, tr[cols], tr["target"])

    ext = {"xgboost": "ubj", "catboost": "cbm", "lightgbm": "txt", "logreg": "joblib"}[a.kind]
    out = MODELS / f"universe_{a.kind}_h{a.horizon}.{ext}"
    if hasattr(m, "save_model"):
        m.save_model(str(out))
    else:
        import joblib
        joblib.dump(m, out)
    print(f"saved -> {out}")
    print(f"\nbacktest it:\n  python backtest_vs_spy.py {out} "
          f"--year {a.test_year} --horizon {a.horizon}")


if __name__ == "__main__":
    main()
