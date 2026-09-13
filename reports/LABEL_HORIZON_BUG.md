# Label-horizon leakage in dataset_v3.parquet (found 2026-07-25)

**Severity: invalidates every metric ever produced from `data/processed/dataset_v3.parquet`.**

Found while reconciling why `train_models.py` reported ~0.43 macro-F1 when the
`experiments/` sweep reported ~0.50. The answer was not the metric convention.

## What was wrong

`dataset_v3.parquet`'s label column was not a 5-day forward return. Brute-forcing the
construction back out of the price panel gives an **exact match, corr = 1.000000**:

```python
fwd_ret = close.shift(-25) / close.shift(-5) - 1.0
```

That is a **20-trading-day forward return starting 5 trading days ahead** — its label
window spans `t+5 … t+25`.

Corroborating evidence, all consistent:

| check | result |
|---|---|
| corr(stored `fwd_ret`, fresh 5-day fwd) | **-0.021** (no relationship) |
| corr(stored `fwd_ret`, fresh 20-day fwd) | 0.739 |
| corr(stored `fwd_ret`, fresh 25-day fwd) | 0.883 |
| corr(stored `fwd_ret`, `shift(-25)/shift(-5)`) | **1.000000** |
| stored `fwd_ret` std | 0.05419 — exactly the fresh **20**-day std |
| corr(stored `ret_5d`, fresh `ret_5d`) | **1.00000** — features and index alignment were fine |

That last row matters: the *features* were correct and perfectly aligned to the current
price panel. Only the label was wrong. Nothing looked broken from the outside.

## Why it leaked

`train_models.py` declares `HORIZON = 5` and purges 5 trading days:

```python
cut = common.purge_cutoff(trading_index, first_test, HORIZON)   # HORIZON = 5
common.assert_no_overlap(trading_index, cut, first_test, HORIZON)
```

The label window ends at `t+25`, so a correct purge needs **25** trading days.

> **20 trading days of label leakage in every walk-forward fold.**

## Why every existing guard missed it

The repo's leakage controls are genuinely good, and none of them could catch this,
because the two halves of the invariant lived in different places:

- the **label horizon** was baked into a parquet file that nobody re-derived
- the **purge horizon** was a constant in a training script

`common.assert_no_overlap` can only verify the horizon it is *handed*. Passing it `5`
made it assert a 5-day claim against a 25-day reality — the assertion passed while
being vacuous. Likewise the feature truncation test, forward-correlation audit and
shuffled-label check all validate *features*, and the features were fine.

The lesson is narrow and reusable: **an invariant that spans a data file and a code
constant must be re-measured from the data, never asserted from the constant.**

## What is affected

Invalid (produced from `dataset_v3.parquet` with a 5-day purge):

- `reports/SR_NEWS_MODEL_RESULTS.md` — the whole results table
- `models/results.json` / `results.md` — the pre-2026-07-25 versions
- the "base (v3-only) reference" numbers 0.4336 (XGBoost) / 0.4360 (CatBoost)
- every `train_models.py` run before 2026-07-25

**Not affected** — `experiments/` and everything downstream of it, including the
production blend. `exp_harness.load_sweep_dataset` builds its label fresh from prices
via `build_label_h(close, cat, HORIZON, SKIP)` and sets `embargo = HORIZON + SKIP`, so
the label horizon and the purge derive from the *same two constants* and cannot
desynchronize. This is the design that should have been used everywhere.

## Fix

1. `etl.py`'s `build_labels` derives the horizon from `config.yaml` (`horizon: 5`, no
   skip). Verified to agree with `common.build_label` on **100%** of rows.
2. `dataset_v3.parquet` rebuilt: 3,357,600 rows, class balance 0.326 / 0.333 / 0.341.
   The previous file is preserved as `dataset_v3_pre_rebuild.parquet`.
3. `tests/test_label_horizon_guard.py` brute-forces `(horizon, skip)` back out of the
   stored `fwd_ret` and fails if it disagrees with `config.yaml`, plus asserts the
   applied purge covers the *measured* label window. The detector is itself tested
   against known (5,0), (10,0) and (25,5) constructions so it can't silently no-op.

Old-vs-new label agreement on shared rows was **0.362** — near random. That is the
expected signature of a *different target*, not a drifted one.

## Impact on the results, measured

Retrained on the corrected label (`train_models.py`, 2026-07-25). Baselines on the
2,662,122 test rows: **uniform random macro-F1 0.3335**, always-majority 0.1696.
(The "≈0.364 majority-class baseline" quoted in the old report is not a valid
macro-F1 reference for a balanced tercile target.)

| model | leaked | corrected | Δ |
|---|---|---|---|
| LSTM +sr+news | 0.4267 | **0.4168** | −0.010 |
| XGBoost +sr | 0.4329 | 0.4162 | −0.017 |
| XGBoost +sr+news | 0.4329 | 0.4161 | −0.017 |
| CatBoost +sr+news | 0.4355 | 0.4123 | −0.023 |
| CatBoost +sr | 0.4351 | 0.4122 | −0.023 |
| LogReg +sr | 0.4252 | 0.3633 | **−0.062** |

Three findings that only appear once the leak is removed:

1. **The leaderboard reorders.** On the leaky label CatBoost led and the LSTM was
   last; corrected, the LSTM is nominally first (0.4168) with XGBoost statistically
   tied (0.4162, a 0.0006 gap) and CatBoost clearly behind (0.4122). The leak
   flattered the tree models and penalized the sequence model.
2. **The leak scaled with model linearity.** LogReg lost 0.062 versus CatBoost's
   0.023 — a linear model reads 20-day overlapping-window autocorrelation almost
   directly. At 0.3633 against a 0.3335 random baseline, logistic regression has
   only marginal skill on the real target.
3. **The regime narrative was partly an artifact.** XGBoost's per-year macro-F1 spread
   collapses from 0.390–0.467 (leaky) to 0.409–0.426 (corrected). The dramatic 2022
   collapse and 2025–26 surge described in `SESSION_SUMMARY.md` were substantially
   properties of the bad label, not of the market. Don't build regime logic on that
   story without re-deriving it.

## Side note: the "mean vs median benchmark" bug never mattered

While verifying the label, one incidental finding: `common.build_label` with
`benchmark='mean'`, `benchmark='median'`, and `etl.build_labels`' leave-one-out mean
all produce **identical terciles** (100% agreement). The benchmark is subtracted from
every fund in a `(date, category)` group before ranking *within that same group*, and
a within-group affine shift cannot change within-group rank order. So the historical
04_baseline "mean vs median" label bug had **zero effect** on the tercile label — it
only ever mattered for the continuous `rel` column.
