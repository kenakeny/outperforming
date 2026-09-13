"""
saudi.py -- US -> Saudi ETF transfer, and the Tadawul benchmark comparison.

Replaces saudi_transfer.py + saudi_transfer_v2.py + saudi_vs_tasi.py (618 lines).

    python saudi.py --build      # download + cache both panels
    python saudi.py              # run the transfer arms, benchmark vs TASI

Arms: saudi_only (trained on Saudi alone) / zeroshot (US model, untouched) /
finetune (US pre-trained, head retrained on Saudi). The LSTM is the only
architecture that transfers -- trees score ~0 zero-shot.
"""
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import core
import etl

warnings.filterwarnings("ignore")
CACHE = pathlib.Path("data/processed/transfer")
CACHE.mkdir(parents=True, exist_ok=True)
TICKERS = [f"{n}.SR" for n in range(9400, 9412)]     # 9412 has too little history
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ, SEEDS = 20, (0, 1, 2)


def build(horizon):
    import yfinance as yf
    df = yf.download(TICKERS, period="max", auto_adjust=True, progress=False,
                     group_by="column", threads=True)
    f = [df[k] for k in ("Close", "High", "Low", "Volume")]
    for x in f:
        x.index.name, x.columns.name = "date", "ticker"
    sa, _ = core.panel(horizon, close=f[0], high=f[1], low=f[2], volume=f[3])
    sa = sa.loc[sa.index.get_level_values("date") >= "2024-01-01"]
    sa.to_parquet(CACHE / "saudi.parquet")

    c, h, l, v, _ = etl._load_panel(etl.load_config())
    c = c.loc[c.index >= "2019-01-01"]
    h, l, v = (x.reindex(index=c.index, columns=c.columns) for x in (h, l, v))
    core.panel(horizon, close=c, high=h, low=l, volume=v)[0].to_parquet(CACHE / "us.parquet")
    print(f"saudi {len(sa):,} rows | us cached")


def net(nf, hidden=48):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(nf, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(32, 3))

        def forward(self, x):
            return self.head(self.lstm(x)[1][0][-1])
    return Net().to(DEV)


def sequences(d, feats, mu, sd, dates=None):
    """(X, y, fwd_ret, date, ticker) per rolling window. Returns tickers because the
    portfolio step needs to know what it holds."""
    mask = {np.datetime64(x, "ns") for x in dates} if dates is not None else None
    X, y, fw, dt, tk = [], [], [], [], []
    for t, g in d.reset_index().sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        f = np.nan_to_num(((g[feats] - mu) / sd).to_numpy(np.float32))
        for j in range(len(g) - SEQ + 1):
            e = j + SEQ - 1
            if mask is not None and g["date"].to_numpy()[e] not in mask:
                continue
            X.append(f[j:e + 1]); y.append(g["target"].to_numpy()[e])
            fw.append(g["fwd_ret"].to_numpy()[e]); dt.append(g["date"].to_numpy()[e]); tk.append(t)
    return (np.stack(X), np.array(y), np.array(fw), np.array(dt), np.array(tk)) if X else None


def train(m, X, y, epochs, lr, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    lossf, Xt, yt = nn.CrossEntropyLoss(), torch.tensor(X, device=DEV), torch.tensor(y, device=DEV, dtype=torch.long)
    m.train()
    for _ in range(epochs):
        idx = torch.randperm(len(Xt), device=DEV)
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            opt.zero_grad(); lossf(m(Xt[b]), yt[b]).backward(); opt.step()
    return m


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(m(torch.tensor(X[i:i + 512], device=DEV)), 1)
                               .cpu().numpy() for i in range(0, len(X), 512)])


def folds(dates, n=4, embargo=5):
    """embargo (trading days) must be >= the label horizon: the label at date t
    reads price data through t+horizon, so any train row closer than `horizon`
    days to the test block leaks test-period prices into its label."""
    u = np.array(sorted(pd.Index(dates).unique()))
    for blk in np.array_split(np.arange(len(u)), n + 1)[1:]:
        yield u[:max(blk[0] - embargo, 0)], u[blk]


def benchmarks(dates, horizon):
    import yfinance as yf
    out = {}
    for tk, name in (("9400.SR", "FALCOM Saudi Equity"), ("^TASI.SR", "TASI")):
        s = yf.download(tk, start="2023-01-01", auto_adjust=True, progress=False)["Close"]
        s = (s.iloc[:, 0] if hasattr(s, "columns") else s).dropna()
        f = s.shift(-horizon) / s - 1.0
        out[name] = np.array([f.iloc[min(s.index.searchsorted(pd.Timestamp(d)), len(s) - 1)]
                              for d in dates], dtype=float)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=15.0)
    a = p.parse_args()
    if a.build:
        return build(a.horizon)

    sa, us = pd.read_parquet(CACHE / "saudi.parquet"), pd.read_parquet(CACHE / "us.parquet")
    feats = [c for c in sa.columns if c not in core.LABEL_COLS]
    print(f"saudi {len(sa):,} | us {len(us):,} | {len(feats)} features\n")

    mu, sd = us[feats].mean(), us[feats].std().replace(0, 1)
    s = sequences(us.sample(min(len(us), 400_000), random_state=0), feats, mu, sd)
    pre = {k: v.clone() for k, v in
           train(net(len(feats)), s[0], s[1], 3, 1e-3).state_dict().items()}
    print(f"pre-trained on {len(s[0]):,} US sequences\n")

    dates = sa.index.get_level_values("date")
    arms = {"saudi_only": [], "zeroshot": [], "zeroshot_pure": [], "finetune": []}
    for i, (tr_d, te_d) in enumerate(folds(dates, embargo=a.horizon)):
        tr, te = sa[dates.isin(tr_d)], sa[dates.isin(te_d)]
        smu, ssd = tr[feats].mean(), tr[feats].std().replace(0, 1)
        trs, tes = (sequences(sa, feats, smu, ssd, d) for d in (tr_d, te_d))
        # zeroshot_pure standardizes with the US mean/std the net was trained on,
        # so NO Saudi data touches the model at any point -- not even to centre
        # the features. `zeroshot` uses Saudi fold statistics, which is already a
        # mild form of domain adaptation.
        tes_us = sequences(sa, feats, mu, sd, te_d)
        if trs is None or tes is None:
            continue
        print(f"fold {i}: {len(trs[0]):,} -> {len(tes[0]):,} sequences", flush=True)
        for arm, init, epochs, lr, freeze in (("saudi_only", None, 30, 1e-3, False),
                                              ("zeroshot", pre, 0, 1e-3, False),
                                              ("zeroshot_pure", pre, 0, 1e-3, False),
                                              ("finetune", pre, 30, 3e-4, True)):
            ps = []
            for seed in SEEDS:
                m = net(len(feats))
                if init:
                    m.load_state_dict(init)
                if freeze:
                    for q in m.lstm.parameters():
                        q.requires_grad = False
                t = tes_us if arm == "zeroshot_pure" else tes
                ps.append(predict(train(m, trs[0], trs[1], epochs, lr, seed) if epochs else m,
                                  t[0]))
            t = tes_us if arm == "zeroshot_pure" else tes
            pr = np.mean(ps, axis=0)
            arms[arm].append(pd.DataFrame({"date": pd.to_datetime(t[3]), "ticker": t[4],
                                           "score": pr[:, 2] - pr[:, 0], "fwd_ret": t[2],
                                           "rel": t[2], "adv": 1.0,
                                           "target": t[1], "pred": pr.argmax(1)}))

    print(f"\n{'series':22} {'ann':>9} {'IR':>7} {'t':>7}")
    curves = {}
    for arm, parts in arms.items():
        bt = core.backtest(pd.concat(parts, ignore_index=True), a.horizon, top=0.33,
                           cost_bps=a.cost_bps)
        curves[arm] = bt
        st = core.stats(bt, a.horizon, col="port")
        print(f"{arm:22} {st['ann']:+9.2%} {st['ir']:+7.2f} {st['t']:+7.2f}")

    bt = curves["finetune"]
    py = 252 / a.horizon
    print(f"{'EW 12 ETFs':22} {bt['bench'].mean()*py:+9.2%}")
    for name, r in benchmarks(bt["date"], a.horizon).items():
        r = r[~np.isnan(r)]
        print(f"{name:22} {r.mean()*py:+9.2%} {r.mean()*py/(r.std()*np.sqrt(py)):+7.2f}")

    # ---- confusion matrices (row-normalized) --------------------------------
    lab = ["under", "neutral", "over"]
    preds = {}
    for arm, parts in arms.items():
        p = pd.concat(parts, ignore_index=True)
        preds[arm] = p
        m = np.zeros((3, 3))
        for tt, pp in zip(p["target"], p["pred"]):
            m[int(tt), int(pp)] += 1
        n = m / np.clip(m.sum(1, keepdims=True), 1, None)
        flip = (n[0, 2] + n[2, 0]) / 2
        share = m.sum(0) / m.sum()
        print(f"\n=== {arm} (n={len(p):,}) ===")
        print("            pred_under  pred_neut  pred_over")
        for i, c in enumerate(lab):
            print(f"  true_{c:<8}" + "  ".join(f"{v:10.3f}" for v in n[i]))
        print(f"  predicted share: under={share[0]:.3f} neutral={share[1]:.3f} over={share[2]:.3f}")
        print(f"  tail flip rate (under<->over): {flip:.3f}   "
              f"accuracy: {(p['target'] == p['pred']).mean():.3f}")
    pd.concat(preds, names=["arm"]).to_parquet("reports/saudi_predictions.parquet")
    print("\npredictions -> reports/saudi_predictions.parquet")


if __name__ == "__main__":
    main()
