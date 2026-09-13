# What else can we possibly do — plan as of 2026-07-27

## Where we actually are

| finding | confidence |
|---|---|
| Liquidity screening is the dominant lever (ρ=0.914, 4 independent confirmations) | **established** |
| Model does **asset-class rotation**, not security selection (coarse-label test: IC 0.080 → 0.023) | **established** |
| Transfer learning works, LSTM only (Saudi Sharpe: 0.39 → 0.85 → 1.22) | **established** |
| Signal is significant at daily-IC level (t=2.87) but **portfolio wrapper loses it** (t=1.44) | **established** |
| News adds nothing (3 independent tests) | **established** |
| No US variant beats SPY risk-adjusted (best Sharpe 0.84 vs 0.96) | **established** |
| Saudi beats TASI by 16.5pp, but 9.9pp is diversification and it dies at 65bps | **established** |
| Screened US edge is genuinely positive | **DSR 0.75–0.89 — likely, not proven** |
| Concentration / hysteresis tuning helps | **refuted (noise)** |

The single most actionable number in that table is **t = 2.87 → 1.44**. The signal exists;
the way we turn it into a portfolio destroys roughly half its statistical content. Nothing
in Tier 1 below requires a better model.

---

## TIER 1 — direct evidence these would help

### 1.1 Portfolio construction overhaul  ⭐ highest expected value
**Why:** three separate pieces of evidence say the wrapper is lossy.
- signal t=2.87, portfolio t=1.44
- screened tilt Sharpe 0.75 vs its own equal-weight universe 0.78 — the tilt adds return
  *and proportionally more volatility*, so risk-adjusted it adds nothing
- top-N cutoff discards the entire ranking below the threshold; the audit showed the
  cutoff level itself is noise (top5/10/20/33 all equivalent), which is exactly what you
  see when the cutoff is the wrong primitive

**What:** replace `top-N equal weight` with, in order of expected payoff:
1. **inverse-volatility weighting** inside the basket (directly targets the Sharpe leak)
2. **score-proportional / rank weighting** (uses the full cross-section, no cutoff)
3. **score shrinkage** toward zero before weighting (the model is overconfident — see
   confusion matrices)
4. **turnover-aware trading** — trade only when expected gain > round-trip cost

**Cost:** ~1 hour. **Reuses saved score parquets — no retraining, no new model search**,
so it barely adds to the multiple-testing burden.
**Success =** portfolio t-stat moves materially toward the signal's 2.87, and Sharpe
exceeds the equal-weight universe rather than matching it.

### 1.2 Market-neutral framing — dissolves the SPY problem
**Why:** the entire "can't beat SPY" result assumes the product competes *with* equity
beta. A market-neutral or beta-hedged sleeve doesn't need to beat SPY — it needs to be
**uncorrelated**, which is a different and much lower bar. The models' per-year excess vs
SPY is strongly negatively correlated with the mega-cap regime (win 2022 and 2025–26,
lose 2021/2023) — that is exactly the profile of a useful diversifier.

For a robo-advisor this is the honest product: a satellite sleeve with low correlation to
the core, not an index substitute.

**What:** long top decile / short bottom decile, or long-only with a SPY beta hedge.
Measure correlation to SPY, and the Sharpe of a 90/10 SPY+sleeve blend vs 100% SPY.
**Cost:** ~1 hour, reuses saved scores.
**Success =** blend Sharpe > SPY Sharpe, even if standalone Sharpe is lower.

### 1.3 Saudi at a 20-day horizon
**Why:** the Saudi work ran at **5-day** rebalancing (`saudi_transfer.HORIZON = 5`), paying
costs ~50×/year. It dies at 35–65bps, and Tadawul's Tier A cap is 65bps. A 20-day version
cuts cost drag ~4×, moving breakeven from ~15bps to ~60bps — from "fails at the regulatory
ceiling" to "marginal at it". The same change was decisive in the US.
**Cost:** ~1 hour (rebuild Saudi panel at h=20, rerun the three LSTM arms).
**Success =** positive net return at 65bps.

---

## TIER 2 — plausible, untested, more expensive

### 2.1 Build the asset-class rotation product explicitly
**Why:** the coarse-category test proved the skill is *across* asset classes, not within
them. Yet we keep asking the model to rank 400–1,580 funds, most of which are near-duplicates.
Give it the job it's actually good at: rotate among ~15–25 highly liquid asset-class proxies
(SPY, QQQ, IWM, EFA, EEM, AGG, TLT, HYG, GLD, DBC, VNQ, XL* sectors).

Solves capacity, cost (all sub-3bps spreads), and explainability simultaneously — and it is
a normal, saleable robo-advisor feature.
**Cost:** ~3 hours (new universe, rebuild features, retrain).
**Risk:** small cross-section (~20) may be too thin to rank; the Saudi work suggests it can
work at n=12, but that result is cost-fragile.

### 2.2 Benchmark-relative label
**Why:** the model is trained to predict excess vs the *universe mean* but judged on excess
vs *SPY*. That mismatch is the direct cause of the unintended cap-weighting bet. Train the
label as excess vs SPY (US) / vs TASI (Saudi).
**Cost:** ~2 hours. **Success =** positive alpha vs SPY, not just vs the universe.

### 2.3 Ensemble across models and horizons
**Why:** XGB and CatBoost disagree sharply year-by-year (2021: XGB −22%, CAT +1%) yet
converge to similar totals — the signature of diversifiable model error. Averaging should
raise Sharpe with no new signal. The repo's production blend already does this for
XGB+CatGNN; it has never been redone post-label-fix.
**Cost:** ~30 min, reuses saved scores.

### 2.4 Selectivity / abstention
**Why:** the confusion matrices show the model detects *extremeness* well and *direction*
poorly (~52% on tails). Trading only high-confidence names and holding the benchmark
otherwise was tried pre-label-fix (`experiments/selectivity_sweep.py`) and is worth redoing
on correct data.
**Cost:** ~1 hour, reuses saved scores.

---

## TIER 3 — rigor and infrastructure (do regardless of the above)

- **DSR + CPCV on the Saudi result.** It is now the strongest claim in the repo
  (Sharpe 1.22) and has never been deflated. It is also the smallest sample. This is the
  claim most likely to be overfit and least audited.
- **Real Saudi spreads** via SAHMK market-depth (~$149/mo). Settles the single assumption
  the whole Saudi case rests on. Everything else there is guesswork until this exists.
- **Revisit `window_years: 1`** in `experiments/eval_production.json` — chosen on the
  leaky label; expanding-window now scores ≥ rolling-1y.
- **Wire the LSTM into `inference.py`** — it is the best transfer model and currently
  unservable.
- **Model card / governance pack** for the fiduciary use case: documented validation,
  DSR figures, known limitations, the label-bug post-mortem.

---

## NOT worth doing (evidence says stop)

- More news features — 3 tests, all neutral-to-negative, importance ranks near-last
- Tuning concentration / hysteresis / top-N — variance share 0.7%, pure noise
- More price-derived features on the existing matrix — `SESSION_SUMMARY` already
  concluded near the noise floor, and the label fix didn't change that
- Chasing the direction problem with more classifier tricks — ~8 approaches, all ~55%
- Extending US history backwards — survivorship bias is unbounded and unfixable from
  `financedatabase` (dead funds simply absent)
- GCC expansion via yfinance — UAE unreachable, Qatar adds 2 funds

---

## The discipline constraint

Roughly **20+ configurations** have now been evaluated on 2019–2026. Every further test
inflates the best observed result. Two rules for whatever comes next:

1. **Pre-register.** Write down the test and its success criterion *before* running it.
   Tier 1 above is already written this way.
2. **Deflate at the end.** Re-run `overfit_audit.py` including the new trials. A result
   that doesn't survive DSR with the updated trial count doesn't count.

Tier 1 is deliberately built from tests that **reuse saved scores and change only the
portfolio wrapper** — they explore a different axis than the model search that has been
exhausted, so they add little to the selection burden while attacking the one gap the
evidence actually points at.
