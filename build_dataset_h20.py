"""build_dataset_h20.py -- a 20-trading-day-horizon sibling of dataset_v3.parquet.

Every feature in dataset_v3 is backward-looking and horizon-independent; only
the label (`fwd_ret`/`rel`/`target`) depends on the horizon. So this doesn't
re-run feature engineering at all -- it rebuilds just the label with
etl.build_labels(horizon=20) and rejoins it against the *existing*
features_v3.parquet, exactly mirroring etl.py's own stage_labels + stage_assemble.

config.yaml's labels.horizon stays at 5 throughout -- deliberately not touched,
so nothing else that reads it (dataset_v3.parquet, the 6 production models) is
put at any risk of drifting out of sync with what actually trained them.

    python build_dataset_h20.py

Writes data/processed/labels_h20.parquet and data/processed/dataset_v3_h20.parquet.
"""
import time

import numpy as np
import pandas as pd

import etl

HORIZON = 20


def main():
    cfg = etl.load_config()
    print(f"config labels.horizon = {cfg['labels']['horizon']} (untouched); building horizon={HORIZON} separately")

    t0 = time.time()
    close, high, low, volume, cat = etl._load_panel(cfg)
    print(f"panel loaded: {close.shape} ({time.time()-t0:.0f}s)")

    labels = etl.build_labels(close, cat, HORIZON, cfg["labels"]["min_group"])
    labels.to_parquet("data/processed/labels_h20.parquet")
    print(f"labels_h20: {len(labels):,} rows, {int(labels['target'].notna().sum()):,} labelled "
         f"({time.time()-t0:.0f}s)")

    feats = pd.read_parquet("data/processed/features_v3.parquet")
    d = feats.join(labels, how="inner").dropna(subset=["target"])
    d["target"] = d["target"].astype(np.int8)
    d.to_parquet("data/processed/dataset_v3_h20.parquet")

    print(f"\ndataset_v3_h20: {len(d):,} rows, {len(etl.FEATURE_COLS)} features "
         f"({time.time()-t0:.0f}s total)")
    print("class balance:", d["target"].value_counts(normalize=True).round(4).to_dict())
    print("date range:", d.index.get_level_values("date").min(), "to",
         d.index.get_level_values("date").max())


if __name__ == "__main__":
    main()
