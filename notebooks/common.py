"""
Shared label / eval helpers for notebooks 01-04.

These were previously copy-pasted, with drift, across 04_baseline / 05_model /
06_features_v2 / Untitled.ipynb -- that drift is exactly how the label bug
happened (04_baseline benchmarked against the category MEAN while every other
notebook used the category MEDIAN, so its "baseline" scored a different
target than everything compared against it). One implementation now,
imported everywhere, so it can't happen again.
"""
import pathlib

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# feature groups -- single source of truth for notebooks 03 and 04, so the
# group definitions can't drift apart the way the label benchmark once did.
# ---------------------------------------------------------------------------

TECH = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'vol_20d', 'px_to_sma_10', 'px_to_sma_20', 'px_to_sma_50',
        'px_to_ema_10', 'px_to_ema_20', 'px_to_ema_50', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_pctb', 'bb_bandwidth']
FRIEND = ['ret_20d_skip5', 'beta_60d', 'rel_str_5d']
POS = ['cs_rank_ret_1d', 'cs_rank_ret_5d', 'cs_rank_skip_mom', 'cs_rank_vol_20d']
ALPHA = ['resid_ret_5d', 'resid_ret_20d', 'corr_cat_60d', 'idio_vol_20d', 'rel_vol_20d', 'tstat_rel_5d', 'beat_rate_20d']
CTX = ['px_to_52w_high', 'ret_60d_skip5', 'cat_disp_5d', 'cat_ret_5d', 'cat_n']
FENG = ['peer_rank_20d', 'vol_20d_raw']                          # feature_eng.py's additions
XSELF_BASE = ['beta_60d', 'resid_ret_5d', 'resid_ret_20d', 'corr_cat_60d', 'idio_vol_20d',
              'tstat_rel_5d', 'beat_rate_20d', 'cat_ret_5d']      # mean-derived features with a
XSELF = [f'{n}_xself' for n in XSELF_BASE]                        # leave-one-out variant
VOL = ['rel_volume_20d', 'cs_rank_dollar_vol', 'vol_zscore']      # from Untitled.ipynb

D36 = TECH + FRIEND + POS + ALPHA + CTX
FEATURE_COLS = D36 + FENG + XSELF + VOL


def resolve_root():
    root = pathlib.Path.cwd()
    if root.name == 'notebooks':
        root = root.parent
    return root


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def _tercile_label(excess, cat, min_group):
    ex = excess.stack().rename('excess').reset_index()
    ex.columns = ['date', 'ticker', 'excess']
    ex['cat'] = ex['ticker'].map(cat)
    g = ex.groupby(['date', 'cat'])['excess']
    pct = g.rank(pct=True)
    size = g.transform('size')
    lab = np.select([pct <= 1 / 3, pct <= 2 / 3], [0, 1], default=2).astype(float)
    lab[size < min_group] = np.nan
    ex['label'] = lab
    return ex.set_index(['date', 'ticker'])['label']


def build_label(prices, cat, horizon, min_group, benchmark='median'):
    """Tercile of excess forward-`horizon`-day return, ranked within (date, category).
    Auto-balances to ~1/3 each by construction. Categories with fewer than
    `min_group` members that day get NaN (not a meaningful tercile).

    benchmark='median' (default -- matches 4 of the 5 pre-existing implementations,
    and is more robust to the single-outlier problem in thin categories) or 'mean'
    (04_baseline's original construction -- kept only so notebook 02 can show the
    before/after diff; do not use 'mean' for real modeling).

    Returns a Series indexed by (date, ticker), values in {0.0, 1.0, 2.0} =
    under/neutral/over-perform, NaN where there weren't enough peers that day.
    """
    fwd = prices.shift(-horizon) / prices - 1.0
    cat_bench = fwd.T.groupby(cat).transform(benchmark).T
    excess = fwd - cat_bench
    return _tercile_label(excess, cat, min_group)


# ---------------------------------------------------------------------------
# purge / walk-forward
# ---------------------------------------------------------------------------

def purge_cutoff(trading_index, test_start, h):
    """Last train date whose forward-h label window ends on/before test_start,
    measured in actual trading days on the price panel (not calendar or
    BusinessDay offsets, which undercount around holiday clusters)."""
    pos = trading_index.searchsorted(pd.Timestamp(test_start))
    return trading_index[max(pos - h, 0)]


def assert_no_overlap(trading_index, cut, first_test_date, h):
    pos_last_train = trading_index.searchsorted(cut)
    label_end = trading_index[min(pos_last_train + h, len(trading_index) - 1)]
    assert label_end <= first_test_date, (
        f'train label window ({label_end.date()}) overlaps test '
        f'({first_test_date.date()}) - leak!'
    )


def predict_labels(model, X):
    """Normalize .predict() output across model libraries (CatBoost returns a
    (n,1) column, XGBoost returns (n,)) into a flat int array."""
    return np.asarray(model.predict(X)).ravel().astype(int)


def walk_forward(d, cols, fit_fn, years, horizon, trading_index, label_col='label'):
    """Expanding-window purged walk-forward over `years`.

    fit_fn(train_df, cols) -> fitted model (anything with .predict()). Model-specific
    (xgboost vs catboost, hyperparameters, internal validation split) lives entirely
    in fit_fn -- this function only handles the purge/split/scoring, which must be
    identical regardless of model choice.
    d: long dataframe with a 'date' column and `label_col`.

    Returns (per-year metrics DataFrame, pooled out-of-fold predictions DataFrame,
    the last fold's fitted model).
    """
    from sklearn.metrics import accuracy_score, f1_score

    rows, oof, last_model = [], [], None
    for year in years:
        test_start, test_end = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31')
        te = d[(d['date'] >= test_start) & (d['date'] <= test_end)]
        if te.empty:
            continue
        first_test = te['date'].min()
        cut = purge_cutoff(trading_index, first_test, horizon)
        assert_no_overlap(trading_index, cut, first_test, horizon)
        tr = d[d['date'] <= cut]

        model = fit_fn(tr, cols)
        pred = predict_labels(model, te[cols])
        acc = accuracy_score(te[label_col], pred)
        mf1 = f1_score(te[label_col], pred, average='macro')

        te_no = te[te['date'].isin(trading_index[::horizon])]
        if len(te_no):
            pred_no = predict_labels(model, te_no[cols])
            acc_no = accuracy_score(te_no[label_col], pred_no)
            mf1_no = f1_score(te_no[label_col], pred_no, average='macro')
        else:
            acc_no = mf1_no = np.nan

        rows.append({'test_year': year, 'n_train': len(tr), 'n_test': len(te),
                     'accuracy': acc, 'macro_f1': mf1,
                     'acc_nonoverlap': acc_no, 'mf1_nonoverlap': mf1_no})
        oof.append(pd.DataFrame({'label': te[label_col].values, 'pred': pred, 'year': year}))
        last_model = model

    metrics = pd.DataFrame(rows)
    pooled = (pd.concat(oof, ignore_index=True) if oof
              else pd.DataFrame(columns=['label', 'pred', 'year']))
    return metrics, pooled, last_model


# ---------------------------------------------------------------------------
# leak audits
# ---------------------------------------------------------------------------

def truncation_check(build_features_fn, prices, feature_names, n_dates=3, seed=0, start=None):
    """Rebuild `feature_names` from prices.loc[:t] at a few random dates and compare
    against the full-panel value at that same date, for every ticker. Proves each
    feature is computable from data <= t by construction (catches lookahead bugs).

    build_features_fn(prices) -> dict[name] -> wide (date x ticker) frame.
    Returns a list of (date, feature) pairs that failed; empty list == pass.
    """
    rng = np.random.default_rng(seed)
    full = build_features_fn(prices)
    dates = prices.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    check_dates = pd.to_datetime(sorted(rng.choice(dates.values, size=n_dates, replace=False)))
    bad = []
    for t in check_dates:
        trunc = build_features_fn(prices.loc[:t])
        for name in feature_names:
            full_row, trunc_row = full[name].loc[t], trunc[name].loc[t]
            both = full_row.notna() & trunc_row.notna()
            same_vals = np.allclose(full_row[both], trunc_row[both], rtol=1e-5, atol=1e-8)
            same_nans = (full_row.isna() == trunc_row.isna()).all()
            if not (same_vals and same_nans):
                bad.append((t.date(), name))
    return bad


def forward_correlation_audit(X, excess, feature_cols):
    """|corr(feature, forward excess return)| for every feature, largest first --
    trailing features should all be small; a spike means a feature secretly peeks
    ahead. `excess` is a Series indexed the same way as X (e.g. labels.parquet's
    'rel' column) -- long format already, not a wide (date x ticker) frame."""
    aud = X.join(excess.rename('excess'), how='inner').dropna(subset=['excess'])
    return aud[feature_cols].corrwith(aud['excess']).abs().sort_values(ascending=False)


def shuffled_label_check(train, test, cols, fit_fn, label_col='label', seed=0, tol=0.05):
    """Permute the label within each date on `train` and refit -- real signal should
    not survive label permutation. Returns the shuffled-label accuracy (expect it
    close to random / the majority-class rate; asserts it isn't meaningfully above
    that, which would mean the harness is wired wrong somewhere)."""
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(seed)
    tr_shuf = train.copy()
    tr_shuf[label_col] = (tr_shuf.groupby('date')[label_col]
                           .transform(lambda s: rng.permutation(s.values)))
    model = fit_fn(tr_shuf, cols)
    pred = predict_labels(model, test[cols])
    acc = accuracy_score(test[label_col], pred)
    majority_rate = test[label_col].value_counts(normalize=True).max()
    assert acc < majority_rate + tol, (
        f'shuffled-label accuracy {acc:.3f} exceeds majority-class rate '
        f'{majority_rate:.3f} by more than {tol} - possible wiring bug'
    )
    return acc


def planted_leak_canary(train, test, cols, fwd_returns, fit_fn, label_col='label',
                         best_honest=0.0, margin=0.05):
    """Add the forward return itself as a feature -- a leak by construction. If
    accuracy doesn't jump well past every legitimate result, the harness can't be
    trusted to catch subtler leaks either. Returns the canary accuracy; this
    feature must never be used for real."""
    from sklearn.metrics import accuracy_score
    fwd_col = fwd_returns.stack().rename('fwd_ret_leak')
    tr = train.join(fwd_col, on=['date', 'ticker'])
    te = test.join(fwd_col, on=['date', 'ticker'])
    canary_cols = cols + ['fwd_ret_leak']
    model = fit_fn(tr, canary_cols)
    pred = predict_labels(model, te[canary_cols])
    acc = accuracy_score(te[label_col], pred)
    assert acc > best_honest + margin, (
        f'canary accuracy {acc:.3f} vs best honest {best_honest:.3f} - '
        'harness failed to detect an obvious leak!'
    )
    return acc
