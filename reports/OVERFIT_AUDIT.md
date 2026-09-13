# Overfitting audit of the US backtest — and a correction to the audit itself

Date: 2026-07-26. Script: `overfit_audit.py`. Data: `reports/us_backtest/scores_xgboost_h20.parquet`
(walk-forward OOS scores, 2019-2026, purge = 20d = label horizon).

## TL;DR

1. The first run of this audit reported **DSR 0.08 ("does not survive") alongside PBO 0.087
   ("acceptable")**. Those cannot both be true. Both were **artifacts of one grid-design error**,
   failing in opposite directions. Corrected: **DSR 0.889, PBO 0.413**.
2. The binding constraint was never selection bias — it was **statistical power**. 94 rebalances
   gives the best config `t = 1.44, p = 0.076`: not significant at 5% *before* any adjustment.
3. Testing the same signal on **1,870 daily cross-sections instead of 94 portfolio returns** gives
   **mean rank IC +0.066, Newey-West t = 2.66, p = 0.008**. The edge is real at the 1% level.
4. So: **the model is fine; the portfolio wrapper is what fails.** A significant signal is being
   converted into an insignificant strategy.

## 1. The grid-design error

`build_grid()` crossed liquidity screen × concentration × hysteresis into 32 "trials" and fed them
to both DSR and PBO. Both tools assume the columns are **exchangeable trials drawn under the null**.
They are not — one dimension is a structural economic choice, not a tuned knob:

| dimension | share of IR variance across the grid |
|---|---|
| `adv` (liquidity screen) | **97.3%** |
| `buf` (hysteresis) | 0.6% |
| `top` (concentration) | 0.2% |

| screen | mean ann IR | min | max |
|---|---|---|---|
| ADV ≥ $0 (none) | **−0.731** | −0.906 | −0.491 |
| ADV ≥ $10M | +0.213 | +0.165 | +0.340 |
| ADV ≥ $50M | +0.385 | +0.283 | +0.500 |
| ADV ≥ $200M | **+0.448** | +0.357 | +0.528 |

Pooling that dimension breaks the two tools in **opposite** directions, which is exactly why they
disagreed:

- **DSR is broken downward.** `sr_variance` is computed across all 32 configs, so it absorbs the
  real −0.73→+0.45 structural gap: `0.01902` instead of `0.00033` for a genuinely exchangeable set
  — **58× inflated**. That pushes the "E[max IR] under the null" threshold to **+1.028 annualized**.
  An IR of 0.53 is then declared noise against a noise threshold that is itself nonsense.
- **PBO is broken upward.** Because `adv200M` configs persistently beat `adv0M` configs both in and
  out of sample, the IS winner is trivially reproducible OOS. PBO = 0.087 is measuring "does
  liquidity screening keep working?" (yes, robustly) and says nothing about the knobs actually
  being selected.

A third problem affects DSR regardless of stratification: the configs are variations on one signal
over overlapping baskets, **mean pairwise correlation 0.892, PC1 = 89.6% of variance → 1.24
effective independent trials, not 32.** The multiple-testing penalty was largely fictitious.

### Corrected, stratified (tuning knobs only, screen held fixed)

| screen | best tuning | ann IR | p (unadj) | null thresh | DSR | PBO |
|---|---|---|---|---|---|---|
| ADV ≥ $0 | top33_buf0.5 | −0.491 | 0.909 | 0.185 | 0.030 | 0.048 |
| ADV ≥ $10M | top33_buf0.5 | +0.340 | 0.178 | 0.081 | 0.748 | 0.321 |
| ADV ≥ $50M | top33_buf0.5 | +0.500 | 0.088 | 0.104 | 0.842 | 0.468 |
| **ADV ≥ $200M** | **top10_buf0.0** | **+0.528** | **0.076** | 0.094 | **0.889** | **0.413** |

The honest reading: the concentration/hysteresis choice **is** moderately overfit (PBO 0.41 is not
comfortable, and the "best" tuning flips to `top33_buf0.5` at three of four screens — evidence that
choice is noise). But the *DSR failure was spurious*.

## 2. Power is the real constraint

`adv200M_top10_buf0.0`, the best config in the grid:

- ann IR **+0.528**, T = **94** non-overlapping 20-day periods (7.5 years)
- `t = 1.443`, one-sided **p = 0.076** — IR standard error is ±0.366
- bootstrap 95% CI **[−0.180, +1.214]**, P(IR ≤ 0) = 0.069
- skew +0.61, kurtosis 5.53 (fat-tailed, so the t-approximation is if anything optimistic)

No selection adjustment can rescue this, because it fails *without* one. Reaching `t = 1.96` at
this IR needs ~173 periods ≈ **13.7 years** of monthly rebalances. That data does not exist.

## 3. The signal itself is significant — the portfolio is what loses it

Testing where the observations actually are: the per-day cross-sectional rank IC of score vs
realized relative return, 1,870 daily cross-sections. Overlapping 20-day forward returns make
consecutive ICs ~0.82 autocorrelated, so the iid t-stat (8.06) is meaningless; Newey-West is required.

| screen | n days | mean IC | t (NW, lag 20) | p | IC > 0 |
|---|---|---|---|---|---|
| ADV ≥ $0 | 1,870 | +0.0643 | 2.64 | 0.0083 | 58.4% |
| ADV ≥ $50M | 1,870 | +0.0784 | **2.87** | **0.0040** | 59.5% |
| ADV ≥ $200M | 1,870 | +0.0661 | 2.66 | 0.0077 | 58.6% |

Robustness of the headline claim (ADV ≥ $200M):

- **NW lag sensitivity** — t = 3.14 (lag 10), 2.66 (20), **2.56 (40, the worst case)**, 2.67 (60),
  2.84 (120), 3.26 (252). Never drops below 2.56; p ≤ 0.0103 everywhere.
- **Moving-block bootstrap** — P(mean IC ≤ 0) = 0.0020 / 0.0028 / 0.0012 at 21 / 63 / 126-day
  blocks; 95% CI entirely positive at every block length (≈ [+0.020, +0.118]).

By year the IC is positive in **7 of 8 years**, the sole exception being 2022 (−0.014, essentially
flat) — consistent across both models and all three screens.

**This is the finding that matters.** The signal clears 1% significance by two independent methods.
The portfolio built from it does not clear 5%. Ranking ~1,900 cross-sections down to one top-10%
basket per month discards nearly all the information and most of the statistical power.

## 4. What is actually established

- **The liquidity screen is the one robust, economically-motivated result.** Unscreened is
  *negative* (−0.73 IR); screened is positive. Spread exceeds the edge on thin names, exactly as
  `us_backtest.py:212-216` argued a priori. This was never a tuned parameter and shouldn't be
  audited as one.
- **The signal has a real, modest cross-sectional edge** (IC ≈ 0.066–0.078, p < 0.01).
- **The `top`/`buffer` choice is not established.** Treat `top10_buf0.0` as arbitrary among the
  ADV-screened configs; the IR differences between them (0.36–0.53) are inside the noise band.
- **The regime narrative is weak.** The +0.50 first-half/second-half Spearman comes with slope
  +1.87 and intercept +1.07 — nearly every config improved in the second half, so this is a level
  shift, not evidence that in-sample ranking transfers. The grid average is positive in only 4 of
  8 years.

## 5. What to do next

1. **Fix portfolio construction, not the model.** The gap between IC significance and portfolio
   insignificance is the whole problem. Options, in order of expected payoff: IC-weighted or
   rank-weighted positions instead of an equal-weight top-N cut; a long/short or
   dollar-neutral tilt to use the bottom of the distribution (currently thrown away entirely);
   overlapping cohorts (start a new 20-day sleeve each week, average 4 sleeves) to quadruple
   the number of effective observations without changing the holding period.
2. **Report the IC test as the primary evidence** in any writeup; the portfolio IR is a
   product-realism check, not the significance test.
3. **Do not report DSR/PBO on a mixed grid again.** `overfit_audit.py` now detects a dominant
   dimension (>50% of variance) and stratifies automatically, printing the mixed-grid numbers only
   under an explicit "invalid — do not report" label.

## Reproducing

```bash
python overfit_audit.py            # reuses cached grid (seconds)
python overfit_audit.py --rebuild  # re-runs all 32 portfolio constructions
```

Outputs `reports/overfit_audit/audit.json` and `config_returns.csv`.
