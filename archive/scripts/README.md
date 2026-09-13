# Archived scripts

Moved out of the repo root on 2026-07-26. Each was verified to be imported by nothing
and referenced in no README/report before moving, so archiving them breaks no live path.
They read data files directly and have no local-module imports, so they still run from
here if invoked with the repo root as cwd.

**Not archived — `build_dataset.py` stays at the repo root.** It looks superseded by
`etl.py`, but `etl.py:331` does a function-level `import build_dataset` to reuse its
`download_universe()` for the extract stage. A grep for imports at line-start misses
that, because the import is indented inside the function.

| script | superseded by | note |
|---|---|---|
| `train_5d.py` | `train_models.py` | 5-day-horizon trainer, pre-consolidation. |
| `train_basic_catboost.py` | `train_models.py` | |
| `train_ohlcv_raw.py` | `train_models.py` | raw-OHLCV baseline. |
| `train_windows.py` | `train_models.py` | rolling-window variant. |
| `compare_horizons.py` | — | one-off horizon comparison (2026-07-15). Predates the label-horizon fix, so its numbers are affected by the leak in `reports/LABEL_HORIZON_BUG.md`. |
| `vol_absolute_vs_relative.py` | — | one-off volatility-definition study (2026-07-17). |

Kept rather than deleted because they record approaches already tried — useful when
deciding whether an idea is genuinely new.
