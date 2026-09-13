"""
Audit how much of the measured edge survives repeated experimentation.

Across this session ~20 configurations were evaluated against the same 2019-2026
window. That inflates the best observed result by construction, and no amount of
re-running fixes it. This script quantifies the inflation instead of ignoring it,
using two standard tools that need no new data:

1. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014)
   Adjusts an observed Sharpe for (a) the number of trials, (b) sample length,
   (c) skew and kurtosis of the return stream. Answers: what is the probability
   the true Sharpe is above zero, given that I picked the best of N?

2. PROBABILITY OF BACKTEST OVERFITTING via CSCV
   (Bailey, Borwein, Lopez de Prado & Zhu 2015)
   Splits the return series into S blocks, and over every way of splitting those
   blocks half in-sample / half out-of-sample, checks where the IS-best
   configuration ranks OOS. If the IS winner routinely lands below the OOS median,
   the selection process is fitting noise. PBO > 0.5 means the search is worse
   than useless.

Both operate on a MATRIX of configuration return streams, so the grid below is
built to represent the search space actually explored (liquidity screen x
concentration x hysteresis).

CAVEAT THAT DRIVES THE STRUCTURE OF THIS SCRIPT -- both tools assume the columns
are EXCHANGEABLE trials. They are not. The liquidity screen accounts for ~97% of
the IR spread across this grid and is an economic prior, not a tuned knob (below
~$10M/day the spread exceeds the edge). Pooling it into the trial set breaks both
tools, in opposite directions:

  * sr_variance is inflated 58x, pushing the DSR null threshold to an absurd
    ~1.03 annualized, so a real edge "fails" (DSR 0.08);
  * the IS winner becomes trivially reproducible OOS, so PBO "passes" (0.087)
    while saying nothing about the knobs actually being selected.

So the audit is run STRATIFIED by the structural dimension. Two further
corrections: the configurations are ~0.89 correlated (effective independent
trials 1.24, not 32), and with only 94 non-overlapping 20-day periods the
unadjusted t is 1.44 (p=0.076) -- the sample, not the selection, is the binding
constraint. Unadjusted significance is therefore reported FIRST.

    python -m scripts.evaluation.overfit_audit              # cached grid
    python -m scripts.evaluation.overfit_audit --rebuild    # rebuild grid
"""
import argparse
import itertools
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core import backtest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "overfit_audit"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 20
PER_YEAR = 252 / HORIZON
GAMMA = 0.5772156649015329          # Euler-Mascheroni

MIN_ADVS = [0.0, 1e7, 5e7, 2e8]
TOPS = [0.05, 0.10, 0.20, 0.33]
BUFFERS = [0.0, 0.5]


def build_grid():
    """Per-rebalance net excess return for every configuration, on a common index."""
    sc = pd.read_parquet(ROOT / "reports" / "us_backtest" / "scores_xgboost_h20.parquet")
    cols = {}
    for adv, top, buf in itertools.product(MIN_ADVS, TOPS, BUFFERS):
        s = sc if adv == 0 else sc[sc["adv"] >= adv]
        if s.empty:
            continue
        bt = backtest(s, HORIZON, top, buffer=buf)
        if bt.empty or len(bt) < 40:
            continue
        name = f"adv{int(adv/1e6)}M_top{int(top*100)}_buf{buf}"
        cols[name] = pd.Series(bt["excess_net"].values,
                               index=pd.to_datetime(bt["date"].values), name=name)
    m = pd.DataFrame(cols).dropna(how="any")
    return m


# --------------------------------------------------------------------------- #
#  Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #

def expected_max_sharpe(sr_variance, n_trials):
    """E[max SR] under the null that all trials have true SR = 0.
    sr_variance is the variance of the *per-period* Sharpe across trials."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    e = np.euler_gamma
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return np.sqrt(sr_variance) * ((1 - e) * z1 + e * z2)


def deflated_sharpe(returns, sr_variance, n_trials):
    """Probability the true (per-period) Sharpe exceeds the selection threshold."""
    r = np.asarray(returns, dtype=float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)      # non-excess kurtosis
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return dict(sr=sr, sr_ann=sr * np.sqrt(PER_YEAR), sr0=sr0, dsr=np.nan,
                    skew=skew, kurtosis=kurt, T=T)
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    return dict(sr=sr, sr_ann=sr * np.sqrt(PER_YEAR),
                sr0=sr0, sr0_ann=sr0 * np.sqrt(PER_YEAR),
                dsr=float(stats.norm.cdf(z)), skew=float(skew),
                kurtosis=float(kurt), T=T, n_trials=n_trials)


# --------------------------------------------------------------------------- #
#  PBO via combinatorially symmetric cross-validation
# --------------------------------------------------------------------------- #

def pbo_cscv(M, S=10):
    """M: DataFrame (periods x configurations) of returns. Returns PBO diagnostics."""
    T, N = M.shape
    if N < 2:
        return {}
    blocks = np.array_split(np.arange(T), S)
    half = S // 2
    lambdas, is_ranks = [], []
    for combo in itertools.combinations(range(S), half):
        is_idx = np.concatenate([blocks[b] for b in combo])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in combo])
        is_r, oos_r = M.iloc[is_idx], M.iloc[oos_idx]

        def sharpe(df):
            mu, sd = df.mean(), df.std(ddof=1)
            return (mu / sd.replace(0, np.nan)).fillna(-np.inf)

        is_sr, oos_sr = sharpe(is_r), sharpe(oos_r)
        best = is_sr.idxmax()
        # relative rank of the IS winner among OOS results
        rank = oos_sr.rank(pct=True)[best]
        rank = min(max(rank, 1.0 / (N + 1)), 1.0 - 1.0 / (N + 1))
        lambdas.append(np.log(rank / (1.0 - rank)))
        is_ranks.append(is_sr.rank(pct=True)[best])
    lam = np.array(lambdas)
    return {"n_combinations": len(lam), "S": S, "n_configs": N,
            "pbo": float((lam < 0).mean()),
            "median_logit": float(np.median(lam)),
            "median_oos_pct_rank": float(np.median(1 / (1 + np.exp(-lam))))}


def variance_decomposition(sr_ann, M):
    """Which grid dimension drives the spread in IR? A dimension that is a
    structural/economic choice rather than a tuned knob must NOT be pooled into
    the trial set: it inflates sr_variance (breaking DSR) while simultaneously
    making the IS winner trivially reproducible OOS (breaking PBO)."""
    meta = pd.DataFrame([dict(zip(("adv", "top", "buf"), c.split("_")))
                         for c in M.columns], index=M.columns)
    meta["sr_ann"] = sr_ann
    grand, out = meta["sr_ann"].mean(), {}
    ss_total = ((meta["sr_ann"] - grand) ** 2).sum()
    for dim in ("adv", "top", "buf"):
        ss_between = sum(len(g) * (g["sr_ann"].mean() - grand) ** 2
                         for _, g in meta.groupby(dim))
        out[dim] = float(ss_between / ss_total) if ss_total > 0 else np.nan
    return out, meta


def effective_trials(M):
    """DSR's E[max SR] assumes ~independent trials. These configurations are
    variations on one signal over overlapping baskets, so they are anything but.
    Participation ratio of the correlation eigenvalues gives the effective count."""
    C = M.corr()
    off = C.values[np.triu_indices_from(C.values, 1)]
    ev = np.linalg.eigvalsh(C.values)[::-1]
    ev = ev[ev > 0]
    return {"mean_pairwise_corr": float(off.mean()),
            "pc1_variance_share": float(ev[0] / ev.sum()),
            "n_effective": float((ev.sum() ** 2) / (ev ** 2).sum()),
            "n_nominal": int(M.shape[1])}


def plain_inference(r, n_boot=20000, seed=0):
    """Unadjusted significance of one return stream. This is the binding
    constraint here and no multiple-testing correction can loosen it: with T
    non-overlapping periods the IR standard error is sqrt(PER_YEAR/T)."""
    r = np.asarray(r, dtype=float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    t = sr * np.sqrt(T)
    se_ann = np.sqrt(PER_YEAR / T)
    rng = np.random.default_rng(seed)
    boot = np.array([(lambda s: s.mean() / s.std(ddof=1) * np.sqrt(PER_YEAR))
                     (r[rng.integers(0, T, T)]) for _ in range(n_boot)])
    return {"T": T, "ir_ann": float(sr * np.sqrt(PER_YEAR)), "t_stat": float(t),
            "p_one_sided": float(1 - stats.t.cdf(t, df=T - 1)),
            "ir_ann_se": float(se_ann),
            "boot_ci95": [float(np.percentile(boot, 2.5)),
                          float(np.percentile(boot, 97.5))],
            "boot_p_ir_le_0": float((boot <= 0).mean()),
            "skew": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r, fisher=False))}


def nw_tstat(x, lag):
    """Newey-West HAC t-stat for H0: mean(x)=0. Required here because overlapping
    `horizon`-day forward returns make consecutive daily ICs ~0.8 autocorrelated;
    the iid t-stat overstates significance by ~3x."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    var = (e @ e) / n
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * ((e[l:] @ e[:-l]) / n)
    se = np.sqrt(var / n)
    return {"n": n, "mean": float(x.mean()), "se": float(se),
            "t": float(x.mean() / se),
            "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(x.mean() / se)))),
            "t_iid": float(x.mean() / (x.std(ddof=1) / np.sqrt(n))),
            "autocorr1": float(pd.Series(x).autocorr(1))}


def ic_significance(min_advs=(0.0, 5e7, 2e8)):
    """Test the signal where the data actually is: 1,870 daily cross-sections
    rather than 94 portfolio rebalances. The portfolio wrapper discards almost
    all of the cross-sectional information, so it is a badly underpowered place
    to ask whether an edge exists."""
    sc = pd.read_parquet(ROOT / "reports" / "us_backtest" / "scores_xgboost_h20.parquet")
    out = {}
    for adv in min_advs:
        s = sc if adv == 0 else sc[sc["adv"] >= adv]
        ic = (s.groupby("date")[["score", "rel"]]
                .apply(lambda g: g["score"].corr(g["rel"], method="spearman")
                       if len(g) >= 20 else np.nan)
                .dropna())
        r = nw_tstat(ic.values, lag=HORIZON)
        r["frac_days_positive"] = float((ic > 0).mean())
        r["by_year"] = {int(y): round(float(v), 4)
                        for y, v in ic.groupby(ic.index.year).mean().items()}
        out[f"adv{int(adv/1e6)}M"] = r
    return out


def performance_degradation(M):
    """Split-sample check: rank configs on the first half, measure how they do on the
    second. If the relationship is flat or negative, in-sample ranking is not
    informative about out-of-sample behaviour."""
    h = len(M) // 2
    a, b = M.iloc[:h], M.iloc[h:]
    sr_a = (a.mean() / a.std(ddof=1)) * np.sqrt(PER_YEAR)
    sr_b = (b.mean() / b.std(ddof=1)) * np.sqrt(PER_YEAR)
    rho, p = stats.spearmanr(sr_a, sr_b)
    slope, intercept = np.polyfit(sr_a, sr_b, 1)
    return {"spearman": float(rho), "p_value": float(p), "slope": float(slope),
            "intercept": float(intercept),
            "best_is_config": sr_a.idxmax(),
            "best_is_sharpe_1h": float(sr_a.max()),
            "its_sharpe_2h": float(sr_b[sr_a.idxmax()]),
            "best_2h_sharpe": float(sr_b.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-run portfolio construction for all configs instead of "
                         "reusing the cached config_returns.csv")
    args = ap.parse_args()

    cache = OUT / "config_returns.csv"
    if cache.exists() and not args.rebuild:
        M = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"reusing cached grid {cache} (--rebuild to regenerate)")
    else:
        print("building configuration grid (this re-runs the portfolio construction) ...")
        M = build_grid()
    print(f"  {M.shape[1]} configurations x {M.shape[0]} rebalances "
          f"({M.index.min().date()} -> {M.index.max().date()})\n")

    sr_ann = (M.mean() / M.std(ddof=1)) * np.sqrt(PER_YEAR)
    tbl = pd.DataFrame({"ann_excess": M.mean() * PER_YEAR, "sharpe_ann": sr_ann})
    tbl = tbl.sort_values("sharpe_ann", ascending=False)
    print("configurations ranked by annualized IR of net excess return:")
    print(tbl.round(4).to_string())
    print(f"\n  spread of IR across the search space: "
          f"{sr_ann.min():+.3f} .. {sr_ann.max():+.3f}")

    best = tbl.index[0]
    results = {"grid": tbl.round(5).to_dict("index"), "best_config": best}

    print("\n" + "=" * 74)
    print("1. GRID DESIGN CHECK -- are these exchangeable trials?")
    print("=" * 74)
    shares, meta = variance_decomposition(sr_ann, M)
    results["variance_shares"] = shares
    for dim, sh in sorted(shares.items(), key=lambda kv: -kv[1]):
        print(f"    {dim:4} explains {sh:6.1%} of the IR variance across the grid")
    print()
    print(meta.groupby("adv")["sr_ann"].agg(["mean", "min", "max"]).round(3).to_string())
    dominant = max(shares, key=shares.get)
    stratify = shares[dominant] > 0.5
    if stratify:
        print(f"\n  -> '{dominant}' dominates. It is an economic screen, not a tuned"
              f"\n     knob, so pooling it into the trial set is invalid: it inflates"
              f"\n     sr_variance (deflating DSR) AND makes the IS winner trivially"
              f"\n     reproducible OOS (deflating PBO). Auditing STRATIFIED by it.")

    eff = effective_trials(M)
    results["effective_trials"] = eff
    print(f"\n  trial independence: mean pairwise corr {eff['mean_pairwise_corr']:.3f}, "
          f"PC1 {eff['pc1_variance_share']:.1%} of variance")
    print(f"  -> effective independent trials {eff['n_effective']:.2f} "
          f"(nominal {eff['n_nominal']}). The multiplicity penalty is far smaller "
          f"than the config count suggests.")

    print("\n" + "=" * 74)
    print("2. UNADJUSTED SIGNIFICANCE OF THE BEST CONFIG")
    print("=" * 74)
    pi = plain_inference(M[best])
    results["plain_inference"] = pi
    print(f"  {best}")
    print(f"    ann IR {pi['ir_ann']:+.3f}   T = {pi['T']} non-overlapping "
          f"{HORIZON}-day periods ({pi['T']/PER_YEAR:.1f} years)")
    print(f"    t = {pi['t_stat']:.3f}   one-sided p = {pi['p_one_sided']:.4f}   "
          f"(IR standard error ~{pi['ir_ann_se']:.3f})")
    print(f"    bootstrap 95% CI [{pi['boot_ci95'][0]:+.3f}, {pi['boot_ci95'][1]:+.3f}]"
          f"   P(IR<=0) = {pi['boot_p_ir_le_0']:.4f}")
    print(f"    skew {pi['skew']:+.2f}  kurtosis {pi['kurtosis']:.2f}")
    if pi["p_one_sided"] > 0.05:
        print(f"\n  -> NOT significant at 5% BEFORE any selection adjustment. This is"
              f"\n     the binding constraint; DSR/PBO below can only tighten it."
              f"\n     {pi['T']} observations is the problem, not the statistics.")

    print("\n" + "=" * 74)
    print("3. THE SAME SIGNAL, TESTED WHERE THE DATA IS (daily cross-sections)")
    print("=" * 74)
    try:
        ics = ic_significance()
        results["ic_significance"] = ics
        for k, r in ics.items():
            print(f"  {k:8} n_days {r['n']:5}  mean rank IC {r['mean']:+.4f}  "
                  f"t_NW {r['t']:5.2f}  p {r['p_two_sided']:.2e}")
            print(f"  {'':8} (iid t would be {r['t_iid']:.2f} -- wrong, IC autocorr "
                  f"{r['autocorr1']:+.3f} from overlapping windows)")
            print(f"  {'':8} IC>0 on {r['frac_days_positive']:.1%} of days; by year "
                  + " ".join(f"{y}:{v:+.3f}" for y, v in r["by_year"].items()))
        head = ics.get("adv200M", next(iter(ics.values())))
        if head["p_two_sided"] < 0.01:
            print(f"\n  -> the edge IS significant at 1% here (t_NW {head['t']:.2f}) "
                  f"while the\n     portfolio built from it is not significant at 5% "
                  f"(t {pi['t_stat']:.2f}).\n     That gap is a portfolio-construction "
                  f"problem, not a model problem.")
    except FileNotFoundError:
        print("  scores parquet not found -- run us_backtest.py first")

    print("\n" + "=" * 74)
    print("4. DEFLATED SHARPE + PBO, STRATIFIED (tuned knobs only)")
    print("=" * 74)
    strata, rows = {}, []
    groups = meta.groupby(dominant) if stratify else [("all", meta)]
    for key, grp in groups:
        sub = M[list(grp.index)]
        if sub.shape[1] < 2:
            continue
        s_pp = sub.mean() / sub.std(ddof=1)
        b = s_pp.idxmax()
        d = deflated_sharpe(sub[b], float(np.var(s_pp, ddof=1)), sub.shape[1])
        p = pbo_cscv(sub, S=10)
        strata[key] = {"best": b, "dsr": d, "pbo": p,
                       "plain": plain_inference(sub[b], n_boot=2000)}
        rows.append({dominant: key, "n_cfg": sub.shape[1],
                     "best_tuning": b.split("_", 1)[1],
                     "IR_ann": round(float(s_pp[b] * np.sqrt(PER_YEAR)), 3),
                     "p_unadj": round(strata[key]["plain"]["p_one_sided"], 4),
                     "null_thresh": round(d.get("sr0_ann", np.nan), 3),
                     "DSR": round(d["dsr"], 4), "PBO": round(p["pbo"], 3)})
    results["strata"] = strata
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  DSR > 0.95 survives; PBO > 0.5 means the search is worse than useless.")
    if rows:
        w = [r for r in rows if r[dominant] == best.split("_")[0]]
        if w:
            r0 = w[0]
            print(f"\n  headline stratum ({r0[dominant]}): DSR {r0['DSR']:.3f}, "
                  f"PBO {r0['PBO']:.3f} -> the concentration/hysteresis choice is "
                  f"\n  moderately overfit, but the DSR failure reported on the mixed "
                  f"grid was an artifact.")

    print("\n  [for the record] pooling all configurations as one trial set gives:")
    var_all = float(np.var(M.mean() / M.std(ddof=1), ddof=1))
    d_all = deflated_sharpe(M[best], var_all, M.shape[1])
    p_all = pbo_cscv(M, S=10)
    results["mixed_grid_invalid"] = {"dsr": d_all, "pbo": p_all,
                                     "note": "invalid: mixes a structural screen "
                                             "with tuned knobs"}
    print(f"    sr_variance {var_all:.5f} -> null threshold "
          f"{d_all.get('sr0_ann', float('nan')):+.3f} ann, DSR {d_all['dsr']:.4f}, "
          f"PBO {p_all['pbo']:.3f}")
    print("    ^ both numbers are artifacts of the grid design -- do not report them.")

    print("\n" + "=" * 74)
    print("5. PERFORMANCE DEGRADATION (first half -> second half)")
    print("=" * 74)
    g = performance_degradation(M)
    results["degradation"] = g
    print(f"  Spearman(IR_1st_half, IR_2nd_half) = {g['spearman']:+.3f} (p={g['p_value']:.3f})")
    print(f"  regression slope = {g['slope']:+.3f}  "
          f"(<=0 means in-sample ranking carries no OOS information)")
    print(f"  best config in 1st half: {g['best_is_config']}")
    print(f"    its IR   1st half {g['best_is_sharpe_1h']:+.3f}  ->  "
          f"2nd half {g['its_sharpe_2h']:+.3f}")
    print(f"    best available in 2nd half: {g['best_2h_sharpe']:+.3f}")
    if g["slope"] > 1.2:
        print(f"\n  NB slope {g['slope']:+.2f} with intercept {g['intercept']:+.2f}: the"
              f"\n  second half is better for ~every configuration, so this correlation"
              f"\n  is substantially a regime shift, not evidence that IS ranking"
              f"\n  transfers. See the per-year table below.")

    print("\n" + "=" * 74)
    print("6. IS THE EDGE BROAD OR ONE-REGIME?")
    print("=" * 74)
    yr = pd.DataFrame({
        "n": M.groupby(M.index.year)[best].count(),
        "best_ann_excess": M.groupby(M.index.year)[best].mean() * PER_YEAR,
        "best_hit": M.groupby(M.index.year)[best].apply(lambda s: (s > 0).mean()),
        "grid_avg_ann": M.groupby(M.index.year).apply(lambda gg: gg.mean().mean()
                                                      * PER_YEAR)})
    results["by_year"] = yr.round(4).to_dict("index")
    print(yr.round(4).to_string())
    pos = (yr["best_ann_excess"] > 0).sum()
    print(f"\n  best config positive in {pos}/{len(yr)} calendar years; "
          f"grid average positive in {(yr['grid_avg_ann'] > 0).sum()}/{len(yr)}")

    M.to_csv(OUT / "config_returns.csv")
    json.dump(results, open(OUT / "audit.json", "w"), indent=2, default=str)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
