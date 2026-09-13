# Plan: raising directional skill (op/up F1 + money) with existing data

Current honest ceilings (2026, production blend): direction AUC ~0.58 on extremes,
tercile F1_op/F1_up ~0.43/0.44, 90/5/5 strategy ~32% net / Sharpe 1.9.
Every idea below uses ONLY data already in the repo (yfinance OHLCV panel +
financedatabase metadata + the composition snapshot). Success criteria for any
change: beat the recorded benchmarks on `experiments/eval_protocol.py` (2026),
then confirm the money in the 90/5/5 backtest.

## Why there's room left

All 74 features describe a fund's OWN trailing state. Direction vs peers over
2 weeks is largely driven by things we never encoded: what the MARKET is doing
(factor/regime rotation), what a fund's CLOSEST peers did yesterday (lead-lag),
and the intraday/overnight structure of returns. The 2022 collapse also says the
TARGET itself is regime-sensitive. Four of the five Tier-1 items attack exactly
these gaps.

## Tier 1 — highest expected value

**1. Market-state (regime) features + interactions with fund exposures.**
Build daily market-context series from the panel itself, using big liquid ETFs
already in the universe as proxies: equity trend & vol regime (SPY-like funds),
rates level/momentum (TLT/SHY-like → duration factor), credit (HYG vs LQD),
dollar (UUP), gold (GLD), yield-curve slope proxy. Features per fund-day:
the raw market states (shared across funds) PLUS interactions with fund
exposures — beta_60d × equity_trend, duration-proxy (corr to TLT) ×
rates_momentum, etc. Rationale: whether a high-beta fund beats its category
depends on which factor is rotating — this is conditional information no
fund-local feature carries. The 2026 signal run (long duration / short
leveraged equity) shows the model is already implicitly finding one rotation;
give it the explicit state.

**2. Dispersion-normalized target (regime-invariant label).**
Replace/augment `rel` with `rel / cat_disp` (peer-relative excess scaled by that
day's category dispersion). 2022 killed the models because the same feature
patterns mapped to different rel magnitudes in a different vol regime; a
normalized target makes patterns comparable across regimes and may recover the
2022-poisoned data as usable training signal. Cheap: one-line change in
`build_label_h`, retrain, compare.

**3. Overnight/intraday return decomposition.**
Split daily returns into gap (close→open) and intraday (open→close) components:
`ret_overnight_20d`, `ret_intraday_20d`, their ratio, and peer-relative
versions. Well-documented that these carry different information (institutional
vs retail flow); we have OHLC and never used the O.

**4. Peer lead-lag spillover features.**
For each fund: trailing-correlation-weighted average of PEERS' recent returns
(1d/5d), minus own return — "my closest peers moved, I haven't yet."
Also category-level lead-lag: cross-category momentum rank, does the fund's
category follow another category's move. The GNN found value in same-day peer
state; explicit lagged spillover gives the trees the same edge plus timing.

**5. Magnitude-weighted direction training (meta-labeling).**
Stage2 currently treats a +0.2% and a +15% outperformer identically. Weight
direction-training samples by |rel| (`sample_weight=|rel|` in the stage2 fit)
so the model concentrates on the moves that pay. Nearly free to test.

## Tier 2 — worth trying after Tier 1

**6. Money-flow volume features.** Volume-weighted CLV (Chaikin-style
accumulation), up-day vs down-day volume ratio, signed dollar-volume momentum —
the volume × direction interaction is unexplored (volume features exist, but
none are signed by price direction).

**7. Feature deltas (feature momentum).** d/dt of the strongest features:
idio_vol_20d change, beta change, rank momentum of `mom_accel`. A fund whose
idio vol is RISING is different from one sitting at high vol.

**8. Category rotation persistence.** Trailing autocorrelation of category
relative returns — in trending-rotation regimes, category winners repeat; in
mean-reverting regimes they flip. Feeds the regime story with a per-category
number.

## Tier 3 — bigger structural bets (only if Tiers 1-2 plateau)

**9. Regime-conditional models:** split training by market-vol/dispersion
regime (2-3 buckets), train per-regime stage2, route at inference by current
regime. Direct answer to 2022.

**10. Stacking meta-learner:** small logistic/GBM over (p_ext, p_dir, GNN
score, per-family sub-scores, regime features) — lets the combiner learn WHEN
to trust which signal, replacing the fixed 0.3/0.7 blend weight.

**11. GNN v2:** correlation-weighted edges (not just same-category), 60d
trailing correlation as edge weights; add the market-state vector as a global
node. Combines ideas 1+4 architecturally.

## Execution order & verification

1. exp_14_market_state.py, exp_15_overnight.py, exp_16_spillover.py — build as
   standard families (probe each with `run_experiment`, save to
   experiments/features/), following the exp_11/12 pattern.
2. Quick wins in parallel: #5 (sample weights) and #2 (normalized target) are
   config-level changes to the existing two-stage — test immediately.
3. Add new families to SWEEP_FAMILIES, retrain the two-stage + GNN + blend
   (`train_production.py`), run `eval_protocol.py` and the 90/5/5 money
   backtest.
4. Gate: adopt any change only if it improves BOTH direction AUC on extremes
   (>0.58) and the 90/5/5 net Sharpe (>1.9 baseline from label_split_sweep) on
   2026. Record everything in experiments/results/ as usual.
5. Whatever wins: re-run the 2021-2026 walk-forward before trusting it.
