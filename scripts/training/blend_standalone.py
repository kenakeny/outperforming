"""
Standalone ETF outperformance model, everything in one file.

Predicts whether a US ETF will beat its category peers over the next 10 trading
days. Builds 81 features from raw OHLCV, trains a two-stage XGBoost + a
category-graph neural net, blends them, and evaluates on a held-out year.

Consolidates what was 15 files (exp_harness.py + 10 exp_*.py feature scripts +
model_xgboost/model_catgnn/model_blend + eval_protocol).

--------------------------------------------------------------------------
WHAT YOU NEED
--------------------------------------------------------------------------
Two parquet files (152MB + 62KB). Either copy them in, or generate
them yourself with the included build_dataset.py (free, no API key):

  data/raw/market_data.parquet   wide OHLCV panel.
                                 MultiIndex columns (field, ticker) where field
                                 is Open/High/Low/Close/Volume; DatetimeIndex rows.
  data/raw/metadata.parquet      one row per ticker, must have a 'category' column
                                 (the peer group the label ranks within).

    pip install pandas numpy pyarrow scikit-learn xgboost torch

    python -m scripts.training.blend_standalone                  # full run
    python -m scripts.training.blend_standalone --no-gnn         # XGBoost only
    python -m scripts.training.blend_standalone --data-dir path/ # custom data

Runtime ~20 min on a GPU, a few hours on CPU.

--------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------
LABEL   fwd = 10-day forward return
        rel = fwd - (leave-one-out mean of the fund's category peers)
        target = per-day tercile of rel  ->  0 under / 1 neutral / 2 outperform
        A fund is never part of its own benchmark, which is what makes the
        label peer-relative rather than a disguised market-direction call.

SPLIT   Rolling 1-year training window ending 10 trading days before the test
        year starts. That 10-day embargo MUST equal the label horizon -- the
        label at date t peeks 10 days forward, so without it the training data
        overlaps the test window. (A mismatch here silently inflated every
        result in this project for months.)

MODELS  stage1: P(fund is an extreme mover)      -- binary, top/bottom 15%
        stage2: P(top | extreme)                 -- binary, direction
        xgb score = p_extreme * (2*p_direction - 1)
        catgnn: message-passing over the same-category clique, pairwise-margin
                ranking loss. Different error profile from the trees, which is
                why blending them helps.
        blend = w*z(xgb) + (1-w)*z(gnn), both z-scored per day, w picked on
                validation rank IC only -- never on test.

METRIC  rank IC (daily Spearman of score vs realized peer-relative return) is
        the one that matters. macro-F1 is reported for reference but the two
        disagree: a model can classify well and rank badly.

Reference results (2026 holdout): xgb rank IC 0.026, catgnn 0.070,
blend 0.085 at w_xgb=0.3. Those were logged when the feature set was 74
columns; exp_15_overnight (7 more) was adopted afterwards, so expect small
differences from a fresh run.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

HORIZON = 10          # forward-return window, in trading days
EMBARGO = HORIZON     # must equal HORIZON -- see SPLIT above
TEST_YEAR = 2026
WINDOW_YEARS = 1      # rolling train window; beat expanding history in testing
MIN_GROUP = 3         # a category needs this many funds to form a peer group
TRAIN_THR = 0.10      # "extreme" = top/bottom decile, for stage 2
SEED = 42


# ========================================================================
#  DATA
# ========================================================================

def load_universe(data_dir):
    """Wide OHLCV frames, restricted to funds in a real category."""
    md = pd.read_parquet(data_dir / "market_data.parquet")
    md.index = pd.to_datetime(md.index)
    md.index.name = "date"

    meta = pd.read_parquet(data_dir / "metadata.parquet")
    cat = meta["category"].reindex(md["Close"].columns)
    counts = cat.value_counts()
    keep = counts[(counts >= MIN_GROUP) & (counts.index != "Uncategorized")].index
    kept = cat[cat.isin(keep)].index

    fields = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        w = md[f][kept]
        w.columns.name = "ticker"
        fields[f] = w
    print(f"universe: {len(kept)} funds, {cat[kept].nunique()} categories, "
          f"{fields['Close'].index.min().date()} -> {fields['Close'].index.max().date()}")
    return fields, cat[kept]


def stack(df):
    s = df.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return s


def loo_peer_mean(x, cat):
    """Leave-one-out category-peer mean: a fund is never in its own benchmark."""
    s = x.T.groupby(cat).transform("sum").T
    n = x.notna().T.groupby(cat).transform("sum").T
    return (s - x.fillna(0)) / (n - x.notna().astype(int)).replace(0, np.nan)


def cat_rank(x, cat):
    """Cross-sectional percentile rank (0..1) within category, per day."""
    return x.T.groupby(cat).rank(pct=True).T


# ========================================================================
#  FEATURES -- 81 columns, all strictly trailing (data <= t only)
# ========================================================================

def build_features(fields, cat):
    o, h, l, c, v = (fields[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    r1 = c.pct_change(fill_method=None)
    f = {}

    # ---- volatility tails -------------------------------------------------
    vol_20d = r1.rolling(20).std()
    vcr = cat_rank(vol_20d, cat)
    f["vol_20d"] = vol_20d
    f["vol_pctile_1y"] = vol_20d.rolling(252).rank(pct=True)
    f["vol_of_vol_60d"] = vol_20d.rolling(60).std() / vol_20d.rolling(60).mean()
    f["vol_spike"] = r1.rolling(5).std() / r1.rolling(60).std()
    f["vol_cat_rank"] = vcr
    f["vol_top_decile"] = (vcr >= 0.9).astype(float).where(vcr.notna())
    hl = (h - l) / c
    f["hl_spike"] = hl.rolling(5).mean() / hl.rolling(60).mean()

    # ---- peer-relative momentum ------------------------------------------
    for n in (5, 20, 60):
        rn = c.pct_change(n, fill_method=None)
        f[f"ret_{n}d_peerrel"] = rn - loo_peer_mean(rn, cat)
        f[f"ret_{n}d_catrank"] = cat_rank(rn, cat)
    # skip-week momentum: t-25 -> t-5, dropping the reversal week
    rs = c.shift(5) / c.shift(25) - 1.0
    f["ret_20d_skip5_peerrel"] = rs - loo_peer_mean(rs, cat)
    f["mom_consistency_20d"] = cat_rank((r1 > 0).rolling(20).mean(), cat)
    ex1 = r1 - loo_peer_mean(r1, cat)
    f["streak_rel_5d"] = ex1.rolling(5).sum()

    # ---- distance from own extremes --------------------------------------
    hi252, lo252 = c.rolling(252).max(), c.rolling(252).min()
    p52h, p52l = c / hi252 - 1.0, c / lo252 - 1.0
    sma200, sma50 = c / c.rolling(200).mean() - 1.0, c / c.rolling(50).mean() - 1.0
    rng52 = (c - lo252) / (hi252 - lo252)
    f["px_to_52w_high"] = p52h
    f["px_to_52w_high_catrank"] = cat_rank(p52h, cat)
    f["px_to_52w_low"] = p52l
    f["px_to_52w_low_catrank"] = cat_rank(p52l, cat)
    f["px_to_sma_200"] = sma200
    f["px_to_sma_200_catrank"] = cat_rank(sma200, cat)
    f["px_to_sma_200_peerrel"] = sma200 - loo_peer_mean(sma200, cat)
    f["px_to_sma_50_catrank"] = cat_rank(sma50, cat)
    f["range_pos_52w"] = rng52
    f["range_pos_52w_catrank"] = cat_rank(rng52, cat)

    # ---- liquidity / size -------------------------------------------------
    dv = c * v
    dv20 = dv.rolling(20).mean()
    f["dollar_vol_log"] = np.log1p(dv20)
    f["dollar_vol_catrank"] = cat_rank(dv20, cat)
    amihud = (r1.abs() / dv.replace(0, np.nan)).rolling(20).mean()
    f["amihud_20d"] = amihud
    f["amihud_catrank"] = cat_rank(amihud, cat)
    f["volume_trend"] = v.rolling(20).mean() / v.rolling(60).mean() - 1.0
    f["volume_spike_5d"] = v.rolling(5).mean() / v.rolling(60).mean() - 1.0
    z = (v == 0) & c.notna()
    f["zero_vol_frac_20d"] = z.rolling(20).sum() / c.notna().rolling(20).sum()
    f["turnover_vol_20d"] = dv.rolling(20).std() / dv20

    # ---- beta / idiosyncratic --------------------------------------------
    m = loo_peer_mean(r1, cat)
    beta = r1.rolling(60).cov(m) / m.rolling(60).var().replace(0, np.nan)
    idio = ex1.rolling(20).std()
    f["beta_60d"] = beta
    f["beta_catrank"] = cat_rank(beta, cat)
    f["corr_cat_60d"] = r1.rolling(60).corr(m)
    f["resid_ret_5d"] = ex1.rolling(5).sum()
    f["resid_ret_20d"] = ex1.rolling(20).sum()
    f["idio_vol_20d"] = idio
    f["idio_vol_catrank"] = cat_rank(idio, cat)
    f["idio_frac_20d"] = idio / r1.rolling(20).std().replace(0, np.nan)
    f["tstat_rel_5d"] = ex1.rolling(5).mean() / (
        ex1.rolling(5).std().replace(0, np.nan) / np.sqrt(5))

    # ---- category context -------------------------------------------------
    r5, r20 = c.pct_change(5, fill_method=None), c.pct_change(20, fill_method=None)
    f["cat_disp_5d"] = r5.T.groupby(cat).transform("std").T
    f["cat_disp_20d"] = r20.T.groupby(cat).transform("std").T
    f["cat_ret_5d"] = r5.T.groupby(cat).transform("mean").T
    f["cat_ret_20d"] = r20.T.groupby(cat).transform("mean").T
    f["cat_vol_20d"] = vol_20d.T.groupby(cat).transform("mean").T
    f["cat_n"] = c.notna().T.groupby(cat).transform("sum").T
    f["cat_breadth_5d"] = (r5 > 0).where(r5.notna()).T.groupby(cat).transform("mean").T
    f["disp_x_vol_rank"] = vcr * f["cat_disp_5d"]

    # ---- range-based volatility ------------------------------------------
    low_s, open_s, close_s = l.replace(0, np.nan), o.replace(0, np.nan), c.replace(0, np.nan)
    log_hl = np.log((h / low_s).clip(lower=1e-12))
    log_co = np.log((c / open_s).clip(lower=1e-12))
    park = np.sqrt(((log_hl ** 2) / (4 * np.log(2))).rolling(20).mean())
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    f["parkinson_20d"] = park
    f["gk_20d"] = np.sqrt(gk.clip(lower=0).rolling(20).mean())
    f["parkinson_catrank"] = cat_rank(park, cat)
    f["clv_20d"] = ((2 * c - h - l) / (h - l).replace(0, np.nan)).rolling(20).mean()
    gap = o / c.shift(1).replace(0, np.nan) - 1.0
    f["gap_20d"] = gap.rolling(20).mean()
    f["gap_abs_20d"] = gap.abs().rolling(20).mean()
    rel_range = (h - l) / close_s
    f["range_ratio_5_60"] = (rel_range.rolling(5).mean()
                             / rel_range.rolling(60).mean().replace(0, np.nan))
    f["pk_to_ccvol"] = park / vol_20d.replace(0, np.nan)

    # ---- drawdown / downside ---------------------------------------------
    dd60 = c / c.rolling(60).max() - 1.0
    f["dd_60d"] = dd60
    f["dd_252d"] = c / c.rolling(252).max() - 1.0
    W = 60
    vals = c.ffill().to_numpy()
    ds = np.full(vals.shape, np.nan)
    if len(vals) >= W:
        win = np.lib.stride_tricks.sliding_window_view(vals, W, axis=0)
        with np.errstate(invalid="ignore"):
            am = np.nanargmax(np.where(np.isnan(win), -np.inf, win), axis=-1)
        ds[W - 1:] = (W - 1) - am
    f["days_since_high_60d"] = pd.DataFrame(ds, index=c.index,
                                            columns=c.columns).where(c.notna())
    semivol = r1.where(r1 < 0).rolling(20, min_periods=5).std()
    f["downside_share_20d"] = semivol / vol_20d
    f["sortino_20d"] = r1.rolling(20).mean() / semivol
    f["skew_60d"] = r1.rolling(60).skew()
    f["kurt_60d"] = r1.rolling(60).kurt()
    f["dd_catrank"] = cat_rank(dd60, cat)

    # ---- temporal structure ----------------------------------------------
    smom20 = c.pct_change(20, fill_method=None) / (vol_20d * np.sqrt(20))
    f["autocorr1_60d"] = r1.rolling(60).corr(r1.shift(1))
    f["sharpe_mom_20d"] = smom20
    f["sharpe_mom_60d"] = c.pct_change(60, fill_method=None) / (
        r1.rolling(60).std() * np.sqrt(60))
    f["mom_accel_20d"] = r20 - r20.shift(20)
    f["updays_frac_20d"] = ((r1 > 0).astype(float).where(r1.notna())
                            .rolling(20, min_periods=15).mean())
    tgrid = pd.DataFrame(np.tile(np.arange(len(c), dtype=float)[:, None], (1, c.shape[1])),
                         index=c.index, columns=c.columns)
    tc = np.log(c).rolling(20).corr(tgrid)
    f["trend_str_20d"] = tc.pow(2) * np.sign(tc)
    f["sharpe_mom_catrank"] = cat_rank(smom20, cat)

    # ---- overnight vs intraday -------------------------------------------
    on, intra = o / c.shift(1) - 1.0, c / o - 1.0
    on20, id20 = on.rolling(20).sum(), intra.rolling(20).sum()
    denom = on.abs().rolling(20).sum() + intra.abs().rolling(20).sum()
    f["on_ret_20d"] = on20
    f["id_ret_20d"] = id20
    f["on_share_20d"] = on.abs().rolling(20).sum() / denom.replace(0, np.nan)
    f["on_minus_id_20d"] = on20 - id20
    f["on_vol_ratio_20d"] = on.rolling(20).std() / intra.rolling(20).std().replace(0, np.nan)
    f["on_ret_catrank"] = cat_rank(on20, cat)
    f["id_ret_catrank"] = cat_rank(id20, cat)

    X = pd.DataFrame({k: stack(val) for k, val in f.items()}).astype("float32")
    # ratio features emit +/-inf on zero denominators; xgboost hard-errors on inf
    X = X.mask(np.isinf(X))
    print(f"features: {X.shape[1]} columns, {len(X):,} rows")
    return X


# ========================================================================
#  LABEL + SPLIT
# ========================================================================

def build_label(close, cat, horizon=HORIZON):
    fwd = close.shift(-horizon) / close - 1.0
    rel = fwd - loo_peer_mean(fwd, cat)
    fwd_l, rel_l = stack(fwd), stack(rel)
    target = (rel_l.dropna().groupby(level="date")
              .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                           labels=[0, 1, 2])).astype("int8"))
    return (pd.concat({"target": target, "fwd_ret": fwd_l, "rel": rel_l}, axis=1)
            .dropna(subset=["target"]))


def rolling_masks(date_index, trading_index, test_year=TEST_YEAR,
                  years=WINDOW_YEARS, embargo=EMBARGO):
    """(fit, val, test). Train is a rolling `years`-year window ending `embargo`
    trading days before the test year. The embargo is the whole ballgame: the
    label looks `HORIZON` days forward, so a smaller gap leaks the future."""
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    cut = trading_index[max(trading_index.searchsorted(test_start) - embargo, 0)]

    tr = (date_index <= cut) & (date_index >= test_start - pd.DateOffset(years=years))
    te = (date_index >= test_start) & (date_index <= test_end)

    tr_dates = np.sort(date_index[tr].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.90)]
    val_cut = trading_index[max(trading_index.searchsorted(val_start) - embargo, 0)]
    return tr & (date_index <= val_cut), tr & (date_index >= val_start), te


# ========================================================================
#  MODELS
# ========================================================================

def train_xgb(d, features, fit_m, val_m, gpu=True):
    """Two-stage: P(extreme) x directional edge. One multiclass model kept
    confusing under/over with each other; splitting the questions helped."""
    from xgboost import XGBClassifier

    base = dict(n_estimators=3000, tree_method="hist",
                device="cuda" if gpu else "cpu", early_stopping_rounds=150,
                random_state=SEED, verbosity=0)
    day_pct = d.groupby(level="date")["rel"].rank(pct=True)

    y_top = (day_pct >= 1 - TRAIN_THR).astype("int8")
    y_ext = ((day_pct >= 1 - TRAIN_THR) | (day_pct <= TRAIN_THR)).astype("int8")
    y_ext15 = ((day_pct >= 0.85) | (day_pct <= 0.15)).astype("int8")

    ft, vt = fit_m & (y_ext == 1).values, val_m & (y_ext == 1).values
    stage2 = XGBClassifier(**base, max_depth=6, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=5.0, objective="binary:logistic",
                           eval_metric="auc")
    stage2.fit(d.loc[ft, features], y_top[ft],
               eval_set=[(d.loc[vt, features], y_top[vt])], verbose=False)

    yf = y_ext15[fit_m]
    spw = float((yf == 0).sum() / max((yf == 1).sum(), 1))
    stage1 = XGBClassifier(**base, max_depth=8, learning_rate=0.03,
                           min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=5.0, objective="binary:logistic",
                           scale_pos_weight=spw, eval_metric="aucpr")
    stage1.fit(d.loc[fit_m, features], yf,
               eval_set=[(d.loc[val_m, features], y_ext15[val_m])], verbose=False)

    def score(mask):
        p_ext = stage1.predict_proba(d.loc[mask, features])[:, 1]
        p_dir = stage2.predict_proba(d.loc[mask, features])[:, 1]
        return p_ext * (2.0 * p_dir - 1.0)

    return score, (stage1, stage2)


def train_gnn(d, features, cat, fit_m, val_m, te_m, hid=128, epochs=40, patience=8):
    """Message passing over the same-category clique. For a clique, mean
    aggregation == masked category pooling, so this uses scatter ops instead of
    an edge list (a 500-fund category would need 250k edges)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    med = d.loc[fit_m, features].median()
    mu, sd = d.loc[fit_m, features].mean(), d.loc[fit_m, features].std().replace(0, 1.0)
    Xall = ((d[features].fillna(med) - mu) / sd).clip(-5, 5).fillna(0.0).to_numpy("float32")
    cid_all = pd.Series(pd.Categorical(d.index.get_level_values("ticker").map(cat)).codes,
                        index=d.index).to_numpy()
    rel_all = d["rel"].to_numpy("float32")
    dates = d.index.get_level_values("date")
    nf = len(features)

    def days_of(mask):
        idx = np.flatnonzero(mask)
        return [(g["i"].to_numpy(), cid_all[g["i"].to_numpy()])
                for _, g in pd.DataFrame({"i": idx}, index=dates[idx]).groupby(level=0)]

    d_fit, d_val, d_te = days_of(fit_m), days_of(val_m), days_of(te_m)

    def pool(hh, cid, attn):
        e = (hh * attn).sum(-1)
        emax = torch.full((int(cid.max()) + 1,), -torch.inf, device=hh.device)
        emax = emax.scatter_reduce(0, cid, e, reduce="amax")
        a = torch.exp(e - emax[cid])
        den = torch.zeros_like(emax).scatter_add(0, cid, a)
        a = (a / den[cid]).unsqueeze(-1)
        out = torch.zeros(int(cid.max()) + 1, hh.shape[1], device=hh.device)
        return out.index_add(0, cid, hh * a)[cid]

    class CatGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(nf, hid), nn.GELU())
            self.a1 = nn.Parameter(torch.randn(hid) / hid ** 0.5)
            self.a2 = nn.Parameter(torch.randn(hid) / hid ** 0.5)
            self.m1 = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Dropout(0.2))
            self.m2 = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Dropout(0.2))
            self.n1, self.n2 = nn.LayerNorm(hid), nn.LayerNorm(hid)
            self.head = nn.Sequential(nn.Linear(hid + nf, hid), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(hid, 1))

        def forward(self, x, cid):
            z = self.inp(x)
            z = self.n1(z + self.m1(torch.cat([z, pool(z, cid, self.a1)], -1)))
            z = self.n2(z + self.m2(torch.cat([z, pool(z, cid, self.a2)], -1)))
            return self.head(torch.cat([z, x], -1)).squeeze(-1)

    def infer(model, days):
        model.eval()
        out, pos = np.empty(sum(len(i) for i, _ in days), "float32"), 0
        with torch.no_grad():
            for ix, cc in days:
                s = model(torch.from_numpy(Xall[ix]).to(dev),
                          torch.from_numpy(cc.astype(np.int64)).to(dev)).cpu().numpy()
                out[pos:pos + len(ix)] = s
                pos += len(ix)
        return out

    def ic_of(days, scores):
        pos, ics = 0, []
        for ix, _ in days:
            s = scores[pos:pos + len(ix)]
            pos += len(ix)
            ics.append(pd.Series(s).corr(pd.Series(rel_all[ix]), method="spearman"))
        return float(np.nanmean(ics))

    model = CatGNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    best_ic, best_state, since = -np.inf, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for di in rng.permutation(len(d_fit)):
            ix, cc = d_fit[di]
            if len(ix) < 10:
                continue
            s = model(torch.from_numpy(Xall[ix]).to(dev),
                      torch.from_numpy(cc.astype(np.int64)).to(dev))
            n = len(cc)
            i, j = rng.integers(0, n, 8 * n), rng.integers(0, n, 8 * n)
            ok = (cc[i] == cc[j]) & (rel_all[ix][i] != rel_all[ix][j]) & (i != j)
            i, j = i[ok], j[ok]
            if not len(i):
                continue
            sign = torch.from_numpy(
                np.sign(rel_all[ix][i] - rel_all[ix][j]).astype("float32")).to(dev)
            loss = F.relu(0.1 - sign * (s[i] - s[j])).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        va = ic_of(d_val, infer(model, d_val))
        print(f"  gnn ep {ep:02d}  val_ic {va:+.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if va > best_ic:
            best_ic, since = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                print("  early stop")
                break

    model.load_state_dict(best_state)
    return {"val": infer(model, d_val), "test": infer(model, d_te)}, best_ic


# ========================================================================
#  EVALUATION
# ========================================================================

def daily_z(d, mask, s):
    ser = pd.Series(s, index=d.index[mask])
    g = ser.groupby(level="date")
    return ((ser - g.transform("mean")) / g.transform("std").replace(0, 1.0)).to_numpy()


def daily_ic(d, mask, s):
    df = pd.DataFrame({"s": s, "rel": d.loc[mask, "rel"]}, index=d.index[mask])
    return float(df.groupby(level="date")
                 .apply(lambda g: g["s"].corr(g["rel"], method="spearman")).mean())


def feature_importance(features, models, out_path=None):
    """Gain-based importance for stage1 (extreme) and stage2 (direction) XGBoost models."""
    stage1, stage2 = models
    imp1 = stage1.get_booster().get_score(importance_type="gain")
    imp2 = stage2.get_booster().get_score(importance_type="gain")
    df = pd.DataFrame({
        "feature": features,
        "gain_stage1_extreme": [imp1.get(f, 0.0) for f in features],
        "gain_stage2_direction": [imp2.get(f, 0.0) for f in features],
    })
    for col in ("gain_stage1_extreme", "gain_stage2_direction"):
        df[col + "_pct"] = 100 * df[col] / df[col].sum()
    df["gain_combined"] = df["gain_stage1_extreme_pct"] + df["gain_stage2_direction_pct"]
    df = df.sort_values("gain_combined", ascending=False).reset_index(drop=True)

    print("\ntop 20 features by combined gain (stage1 + stage2, %):")
    print(df.head(20).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"saved -> {out_path}")
    return df


def evaluate(name, d, mask, score):
    """rank IC plus the forced-tercile view. NOTE the two F1 conventions in this
    project are not comparable -- this one forces equal per-day terciles, so its
    baseline is 0.333, not the ~0.50 an argmax classifier reports."""
    from sklearn.metrics import accuracy_score, f1_score

    y = d.loc[mask, "target"].to_numpy()
    s = pd.Series(score, index=d.index[mask])
    pred = (s.groupby(level="date")
            .transform(lambda x: pd.qcut(x.rank(method="first"), 3,
                                         labels=[0, 1, 2]))).astype(int).to_numpy()

    ext = y != 1
    dir_acc = float((pred[ext] == y[ext]).mean()) if ext.any() else np.nan
    res = {
        "name": name,
        "n_test": int(mask.sum()),
        "rank_ic": round(daily_ic(d, mask, score), 4),
        "macro_f1_terciles": round(float(f1_score(y, pred, average="macro")), 4),
        "accuracy_terciles": round(float(accuracy_score(y, pred)), 4),
        "dir_acc_on_extremes": round(dir_acc, 4),
        "under_to_over": round(float((pred[y == 0] == 2).mean()), 4),
        "over_to_under": round(float((pred[y == 2] == 0).mean()), 4),
    }
    print(f"\n[{name}]  rank_ic={res['rank_ic']}  macroF1={res['macro_f1_terciles']}  "
          f"dir_acc={res['dir_acc_on_extremes']}  "
          f"flips={res['under_to_over']}/{res['over_to_under']}")
    return res


# ========================================================================
#  MAIN
# ========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="blend_results.json")
    ap.add_argument("--no-gnn", action="store_true", help="XGBoost only")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--importance-out", default="feature_importance.csv")
    a = ap.parse_args()

    fields, cat = load_universe(pathlib.Path(a.data_dir))
    close = fields["Close"]

    X = build_features(fields, cat)
    features = list(X.columns)

    d = X.join(build_label(close, cat), how="inner").dropna(subset=["target"])
    d = d.dropna(subset=features, how="all")
    d["target"] = d["target"].astype("int8")

    dates = d.index.get_level_values("date")
    fit_m, val_m, te_m = rolling_masks(dates, close.index)
    print(f"split: fit={fit_m.sum():,}  val={val_m.sum():,}  test={te_m.sum():,}  "
          f"(embargo {EMBARGO}d == horizon {HORIZON}d)\n")

    print("training XGBoost two-stage ...")
    xgb_score, xgb_models = train_xgb(d, features, fit_m, val_m, gpu=not a.cpu)
    sx_val, sx_te = xgb_score(val_m), xgb_score(te_m)
    results = {"xgb": evaluate("xgb_two_stage", d, te_m, sx_te)}
    feature_importance(features, xgb_models, out_path=a.importance_out)

    if not a.no_gnn:
        print("\ntraining CatGNN ...")
        gnn, gnn_val_ic = train_gnn(d, features, cat, fit_m, val_m, te_m,
                                    epochs=a.epochs)
        results["gnn"] = evaluate("catgnn", d, te_m, gnn["test"])

        zx_v, zg_v = daily_z(d, val_m, sx_val), daily_z(d, val_m, gnn["val"])
        zx_t, zg_t = daily_z(d, te_m, sx_te), daily_z(d, te_m, gnn["test"])

        print("\npicking blend weight on VALIDATION only:")
        best_w, best_ic = 0.0, -np.inf
        for w in np.arange(0.0, 1.01, 0.1):
            ic = daily_ic(d, val_m, w * zx_v + (1 - w) * zg_v)
            print(f"  w_xgb={w:.1f}  val_ic={ic:+.4f}")
            if ic > best_ic:
                best_w, best_ic = w, ic
        print(f"chosen w_xgb={best_w:.1f} (val_ic {best_ic:+.4f})")

        results["blend"] = evaluate("blend_xgb_gnn", d, te_m,
                                    best_w * zx_t + (1 - best_w) * zg_t)
        results["blend"]["w_xgb"] = round(float(best_w), 2)

    results["config"] = {"horizon": HORIZON, "embargo": EMBARGO,
                         "window_years": WINDOW_YEARS, "test_year": TEST_YEAR,
                         "n_features": len(features)}
    pathlib.Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
