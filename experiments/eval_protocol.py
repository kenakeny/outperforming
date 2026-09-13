"""eval_protocol — the one fixed scorecard every direction model goes through.

Any model that produces a per-row bullishness score for the 2026 test rows
gets judged on the same four things, so results can't be cherry-picked:

  1. direction accuracy on TRUE extremes (fixed 15% eval population) --
     the honest "is it still a coin flip" number
  2. full 3-class confusion matrix (counts + row-normalized), classes
     derived by per-day terciles of the score (same convention as the label)
     unless the model supplies its own 3-class predictions
  3. daily rank IC of the score vs realized peer-relative excess
  4. gated calls at K = 3% and 2% per day (conviction = |score - daily
     median|): win rate, mean signed payoff, win/loss ratio

Import and call `evaluate(...)`; running this file as a script trains the
session-best ensemble (2-week horizon, 1y rolling window, tuned stage2,
10% direction-training threshold) through the new exp_harness helpers and
records its scorecard as the benchmark -- which also verifies the Step-0
harness consolidation reproduces the copy-pasted originals.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from exp_harness import EXP

MC_NAMES = ["under", "neutral", "over"]
EVAL_EXTREME_THR = 0.15
GATE_KS = (0.03, 0.02)


def evaluate(name, d, te_mask, score, pred3=None, extra=None, save=True,
             verbose=True):
    """Score a model on the fixed protocol.

    d: frame with 'target' (0/1/2) and 'rel' columns, (date, ticker) index.
    te_mask: boolean mask of test rows (aligned to d).
    score: continuous bullishness score for the test rows (higher = more
      likely to outperform), len == te_mask.sum().
    pred3: optional explicit 3-class predictions; if None, derived as
      per-day terciles of the score.
    """
    te = pd.DataFrame({"s": np.asarray(score, dtype=float),
                       "y": d.loc[te_mask, "target"].to_numpy(),
                       "rel": d.loc[te_mask, "rel"].to_numpy()},
                      index=d.index[te_mask])
    g_day = te.groupby(level="date")

    # -- 1. direction accuracy on true extremes (fixed 15% population)
    day_pct = g_day["rel"].rank(pct=True)
    is_ext = (day_pct >= 1 - EVAL_EXTREME_THR) | (day_pct <= EVAL_EXTREME_THR)
    s_med = g_day["s"].transform("median")
    dir_call = np.where(te["s"] >= s_med, 2, 0)
    ext = te[is_ext]
    dir_acc = float((dir_call[is_ext.to_numpy()] == ext["y"]).mean())

    # -- 2. confusion matrix (per-day tercile classes unless supplied)
    if pred3 is None:
        s_pct = g_day["s"].rank(pct=True)
        pred3 = np.select([s_pct <= 1 / 3, s_pct <= 2 / 3], [0, 1], default=2)
    pred3 = np.asarray(pred3).astype(int)
    cm = confusion_matrix(te["y"], pred3, labels=[0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    u2o = float(cm[0, 2] / cm[0].sum())
    o2u = float(cm[2, 0] / cm[2].sum())

    # -- 3. daily rank IC
    ic = float(g_day.apply(lambda x: x["s"].corr(x["rel"], method="spearman"))
               .mean())

    # -- 4. gated calls
    te["conv"] = (te["s"] - s_med).abs()
    te["dir"] = dir_call
    gated = {}
    for k in GATE_KS:
        picks = g_day.apply(
            lambda x: x.nlargest(max(int(len(x) * k), 1), "conv"))
        picks = picks.droplevel(0) if picks.index.nlevels > 2 else picks
        win = (picks["dir"] == picks["y"]).to_numpy()
        sign = np.where(picks["dir"] == 2, 1, -1)
        payoff = sign * picks["rel"].to_numpy()
        wins, losses = payoff[payoff > 0], payoff[payoff < 0]
        wl = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None
        gated[f"K{int(k * 100)}"] = {
            "n_calls": int(len(picks)),
            "win_rate": round(float(win.mean()), 4),
            "mean_payoff": round(float(payoff.mean()), 5),
            "win_loss_ratio": round(wl, 3) if wl else None,
        }

    res = {
        "name": name, "n_test": int(te_mask.sum()),
        "dir_acc_on_extremes": round(dir_acc, 4),
        "rank_ic": round(ic, 4),
        "under_to_over": round(u2o, 4), "over_to_under": round(o2u, 4),
        "macro_f1_terciles": round(float(f1_score(te["y"], pred3,
                                                  average="macro")), 4),
        "accuracy_terciles": round(float(accuracy_score(te["y"], pred3)), 4),
        "gated": gated,
        "confusion_matrix": cm.tolist(),
        **(extra or {}),
    }

    if verbose:
        print(f"\n================ {name}")
        print(f"dir_acc_on_extremes={res['dir_acc_on_extremes']}  "
              f"rank_ic={res['rank_ic']}  "
              f"under->over={res['under_to_over']}  "
              f"over->under={res['over_to_under']}")
        print(pd.DataFrame(cm, index=MC_NAMES, columns=MC_NAMES).to_string())
        print("row-normalized (recall):")
        print(pd.DataFrame(cmn.round(3), index=MC_NAMES,
                           columns=MC_NAMES).to_string())
        for kname, gk in gated.items():
            print(f"{kname}: n={gk['n_calls']}  win_rate={gk['win_rate']}  "
                  f"payoff={gk['mean_payoff']}  wl={gk['win_loss_ratio']}")
    if save:
        (EXP / "results" / f"eval_{name}.json").write_text(
            json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# benchmark: the session-best ensemble through the consolidated harness
# ---------------------------------------------------------------------------

def run_benchmark():
    from exp_harness import load_sweep_dataset, gpu_lock
    from xgboost import XGBClassifier

    HORIZON, TRAIN_THR = 10, 0.10
    d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(
        horizon=HORIZON, window_years=1)
    print(f"benchmark dataset: fit={fit_m.sum()} val={val_m.sum()} "
          f"test={te_m.sum()}", flush=True)

    day_pct = d.groupby(level="date")["rel"].rank(pct=True)
    y_top_tr = (day_pct >= 1 - TRAIN_THR).astype("int8")
    y_ext_tr = ((day_pct >= 1 - TRAIN_THR) | (day_pct <= TRAIN_THR)).astype("int8")
    y_ext_15 = ((day_pct >= 0.85) | (day_pct <= 0.15)).astype("int8")

    BASE = dict(n_estimators=3000, tree_method="hist", device="cuda",
                early_stopping_rounds=150, random_state=42, verbosity=0)
    TUNED = dict(max_depth=6, learning_rate=0.03, min_child_weight=8,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)

    ft, vt = fit_m & (y_ext_tr == 1).values, val_m & (y_ext_tr == 1).values
    with gpu_lock():
        stage2 = XGBClassifier(**BASE, **TUNED, objective="binary:logistic",
                               eval_metric="auc")
        stage2.fit(d.loc[ft, features], y_top_tr[ft],
                   eval_set=[(d.loc[vt, features], y_top_tr[vt])], verbose=False)

    y_ext_fit = y_ext_15[fit_m]
    spw = float((y_ext_fit == 0).sum() / (y_ext_fit == 1).sum())
    with gpu_lock():
        stage1 = XGBClassifier(**BASE, max_depth=8, learning_rate=0.03,
                               min_child_weight=8, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=5.0,
                               objective="binary:logistic",
                               scale_pos_weight=spw, eval_metric="aucpr")
        stage1.fit(d.loc[fit_m, features], y_ext_fit,
                   eval_set=[(d.loc[val_m, features], y_ext_15[val_m])],
                   verbose=False)

    p_ext = stage1.predict_proba(d.loc[te_m, features])[:, 1]
    p_dir = stage2.predict_proba(d.loc[te_m, features])[:, 1]
    score = p_ext * (2.0 * p_dir - 1.0)
    evaluate("benchmark_ensemble_2wk", d, te_m, score,
             extra={"horizon": HORIZON, "window_years": 1,
                    "train_extreme_thr": TRAIN_THR})


if __name__ == "__main__":
    run_benchmark()
