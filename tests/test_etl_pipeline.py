"""ETL orchestration and data contracts.

Covers the plumbing (stage ordering, freshness, manifest, panel cleaning) and the
schema contract of the artifacts each stage writes, plus one parity check against
the real dataset_v3.parquet -- because etl.py claims to reproduce a file that was
originally produced by a notebook, and that claim needs evidence.
"""
import numpy as np
import pandas as pd
import pytest

import etl


# --------------------------------------------------------------------------- #
#  panel cleaning                                                             #
# --------------------------------------------------------------------------- #

def test_clean_panel_aligns_all_four_frames(synthetic_market_data, synthetic_metadata):
    close, high, low, volume, cat = etl.clean_panel(synthetic_market_data, synthetic_metadata, 3)
    for frame in (high, low, volume):
        assert frame.index.equals(close.index)
        assert list(frame.columns) == list(close.columns)
    assert list(cat.index) == list(close.columns)


def test_clean_panel_drops_leveraged_funds(synthetic_market_data, synthetic_metadata):
    meta = synthetic_metadata.copy()
    levered = meta.index[0]
    meta.loc[levered, "is_leveraged"] = True
    close, *_ = etl.clean_panel(synthetic_market_data, meta, 3)
    assert levered not in close.columns, "a 3x fund's returns aren't peer-comparable"


def test_clean_panel_drops_categories_below_min_group(synthetic_market_data, synthetic_metadata):
    meta = synthetic_metadata.copy()
    meta.loc[meta.index[0], "category"] = "tiny"
    close, _, _, _, cat = etl.clean_panel(synthetic_market_data, meta, 3)
    assert "tiny" not in set(cat), "kept a category too small to form a tercile"


def test_clean_panel_drops_uncategorized(synthetic_market_data, synthetic_metadata):
    meta = synthetic_metadata.copy()
    meta["category"] = ["Uncategorized"] * 4 + list(meta["category"].iloc[4:])
    close, _, _, _, cat = etl.clean_panel(synthetic_market_data, meta, 3)
    assert "Uncategorized" not in set(cat)


# --------------------------------------------------------------------------- #
#  long-format contracts                                                      #
# --------------------------------------------------------------------------- #

def test_features_long_has_the_declared_schema(synthetic_panel):
    p = synthetic_panel
    frames, regime = etl.build_features(p["close"], p["high"], p["low"], p["volume"], p["cat"])
    long = etl.features_to_long(frames, regime)

    assert list(long.columns) == etl.FEATURE_COLS, "column order is part of the contract"
    assert long.index.names == ["date", "ticker"]
    assert long.index.is_monotonic_increasing
    assert not long.index.duplicated().any()


def test_regime_features_are_identical_across_tickers_on_a_date(synthetic_panel):
    """Market-wide series are broadcast by date -- if they ever varied per ticker,
    the join grain would be wrong."""
    p = synthetic_panel
    frames, regime = etl.build_features(p["close"], p["high"], p["low"], p["volume"], p["cat"])
    long = etl.features_to_long(frames, regime)
    on_date = long.xs(long.index.get_level_values("date")[250], level="date")
    for col in etl.REGIME_COLS:
        assert on_date[col].nunique(dropna=True) <= 1, f"{col} varies by ticker"


def test_sr_long_drops_only_all_nan_warmup_rows(synthetic_panel):
    p = synthetic_panel
    long = etl.sr_to_long(etl.build_sr_features(p["close"], p["high"], p["low"]))
    assert len(long.columns) == 11
    assert not long.isna().all(axis=1).any()
    assert np.isfinite(long.replace(np.nan, 0.0).to_numpy()).all(), "inf survived the clean"


def test_features_are_float32(synthetic_panel):
    """The full panel is millions of rows; float64 doubles memory for precision
    that trailing technical indicators don't have anyway."""
    p = synthetic_panel
    frames, regime = etl.build_features(p["close"], p["high"], p["low"], p["volume"], p["cat"])
    long = etl.features_to_long(frames, regime)
    assert (long.dtypes == np.float32).all()


# --------------------------------------------------------------------------- #
#  orchestration                                                              #
# --------------------------------------------------------------------------- #

def test_stage_registry_is_complete():
    assert set(etl.STAGES) == set(etl.STAGE_ORDER) == set(etl.STAGE_OUTPUTS)


def test_parse_stages_sorts_into_dependency_order():
    assert etl.parse_stages("assemble,features") == ["features", "assemble"]
    assert etl.parse_stages("all") == etl.STAGE_ORDER


def test_parse_stages_rejects_unknown_stage():
    with pytest.raises(SystemExit, match="unknown stage"):
        etl.parse_stages("features,nonsense")


def test_freshness_is_invalidated_by_a_newer_upstream(tmp_path, monkeypatch):
    """Rebuilding features must not leave a stale dataset_v3 behind -- the whole
    point of the freshness check is to catch that."""
    outputs = {s: tmp_path / f"{s}.parquet" for s in etl.STAGE_ORDER}
    monkeypatch.setattr(etl, "STAGE_OUTPUTS", outputs)

    for s in etl.STAGE_ORDER:
        outputs[s].write_text("x")
    import os
    import time
    time.sleep(0.01)
    os.utime(outputs["features"], None)   # features rebuilt after assemble

    assert etl.is_fresh("features")
    assert not etl.is_fresh("assemble"), "stale downstream output reported as fresh"


def test_missing_output_is_not_fresh(tmp_path, monkeypatch):
    outputs = {s: tmp_path / f"{s}.parquet" for s in etl.STAGE_ORDER}
    monkeypatch.setattr(etl, "STAGE_OUTPUTS", outputs)
    assert not etl.is_fresh("features")


def test_run_skips_fresh_stages_and_records_a_manifest(tmp_path, monkeypatch):
    outputs = {s: tmp_path / f"{s}.parquet" for s in etl.STAGE_ORDER}
    monkeypatch.setattr(etl, "STAGE_OUTPUTS", outputs)
    monkeypatch.setattr(etl, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(etl, "ROOT", tmp_path)

    calls = []

    def fake(name):
        def _stage(cfg, force=False):
            calls.append(name)
            outputs[name].write_text("x")
            return {"rows": 1}
        return _stage

    monkeypatch.setattr(etl, "STAGES", {s: fake(s) for s in etl.STAGE_ORDER})

    cfg = {"labels": {"horizon": 5, "min_group": 3}}
    manifest = etl.run(["features", "sr"], cfg=cfg)
    assert calls == ["features", "sr"]
    assert set(manifest) == {"features", "sr"}
    assert manifest["features"]["rows"] == 1

    etl.run(["features", "sr"], cfg=cfg)          # second pass: everything is fresh
    assert calls == ["features", "sr"], "a fresh stage was re-run"

    etl.run(["features"], cfg=cfg, force=True)    # --force overrides freshness
    assert calls[-1] == "features"


def test_config_declares_the_horizon_the_pipeline_uses():
    cfg = etl.load_config()
    assert cfg["labels"]["horizon"] == 5
    assert cfg["labels"]["min_group"] >= 3


# --------------------------------------------------------------------------- #
#  parity with the artifact the notebook produced                             #
# --------------------------------------------------------------------------- #

@pytest.mark.requires_data
def test_dataset_v3_matches_the_declared_schema():
    d = pd.read_parquet(etl.PROC / "dataset_v3.parquet")
    assert d.index.names == ["date", "ticker"]
    missing = set(etl.FEATURE_COLS) - set(d.columns)
    assert not missing, f"etl.FEATURE_COLS lists columns dataset_v3 doesn't have: {missing}"
    assert set(etl.LABEL_COLS) <= set(d.columns)
    assert set(d["target"].dropna().unique()) <= {0, 1, 2}


@pytest.mark.requires_data
def test_dataset_v3_has_no_duplicate_rows_per_date_ticker():
    d = pd.read_parquet(etl.PROC / "dataset_v3.parquet", columns=["target"])
    assert not d.index.duplicated().any()


@pytest.mark.requires_data
def test_dataset_v3_label_is_balanced_across_terciles():
    d = pd.read_parquet(etl.PROC / "dataset_v3.parquet", columns=["target"])
    shares = d["target"].value_counts(normalize=True)
    assert shares.max() < 0.45, (
        f"tercile label is unbalanced ({shares.to_dict()}) -- the ~0.364 "
        "majority-class reference in the reports assumes near-balance")
