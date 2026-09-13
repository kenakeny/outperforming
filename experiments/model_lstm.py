"""model_lstm — 2-layer LSTM over the last 20 trading days of 8 daily channels.

Instead of the hand-built trailing aggregates the tree models use, the LSTM
sees the raw daily path: own return, peer-relative return, within-category
rank, intraday range, close-location value, overnight gap, volume z-score,
and drawdown state. Same label / embargo / 2026 holdout as every other model
(masks come from exp_harness.holdout_masks).

Train samples are taken every 5th trading day so the forward-5d label windows
don't overlap (less pseudo-replication, 5x faster epochs); the 2026 test set
stays daily like the other models. Uses CUDA under the gpu lock when
available, otherwise falls back to CPU.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import (load_universe, loo_peer_mean, cat_pct_rank,
                         build_label, holdout_masks, score_and_save, gpu_lock,
                         EXP)

SEQ, N_CH = 20, 8
SEED = 42

# ---------------------------------------------------------------- channels
fields, cat = load_universe()
o, h, l, c, v = (fields[k] for k in ["Open", "High", "Low", "Close", "Volume"])
r1 = c.pct_change(fill_method=None)
logv = np.log1p(v.where(v > 0))
hl_rng = h - l
channels = {
    "r1":      r1,
    "r1_rel":  r1 - loo_peer_mean(r1, cat),
    "rank_r1": cat_pct_rank(r1, cat) - 0.5,
    "hl":      hl_rng / c,
    "clv":     ((c - l) - (h - c)) / hl_rng.where(hl_rng > 0),
    "gap":     o / c.shift(1) - 1.0,
    "volz":    (logv - logv.rolling(60).mean()) / logv.rolling(60).std(),
    "dd60":    c / c.rolling(60).max() - 1.0,
}
assert len(channels) == N_CH
C = np.stack([ch.to_numpy(dtype="float32") for ch in channels.values()], axis=-1)
dates, tickers = c.index, c.columns
print("channel tensor:", C.shape)

# ---------------------------------------------------------------- samples
lbl = build_label(c, cat)
samp = lbl.reset_index()
samp["di"] = dates.get_indexer(samp["date"])
samp["ti"] = pd.Index(tickers).get_indexer(samp["ticker"])
samp = samp[(samp["di"] >= SEQ - 1) & (samp["ti"] >= 0)]
samp = samp.set_index(["date", "ticker"])

date_index = samp.index.get_level_values("date")
fit_mask, val_mask, te_mask = holdout_masks(date_index, dates)
nonoverlap = date_index.isin(dates[::5])
fit_mask, val_mask = fit_mask & nonoverlap, val_mask & nonoverlap
print(f"samples: fit={fit_mask.sum()}, val={val_mask.sum()}, test={te_mask.sum()}")

# standardize each channel on fit-period stats only; NaN -> 0 (= channel mean)
fit_cut = date_index[fit_mask].max()
Cfit = C[: dates.searchsorted(fit_cut) + 1]
mu = np.nanmean(Cfit, axis=(0, 1))
sd = np.nanstd(Cfit, axis=(0, 1))
sd[sd == 0] = 1.0
Cn = np.clip((C - mu) / sd, -5, 5)
Cn = np.nan_to_num(Cn, nan=0.0).astype("float32")
del C, Cfit

def gather(mask):
    di = samp.loc[mask, "di"].to_numpy()
    ti = samp.loc[mask, "ti"].to_numpy()
    y = samp.loc[mask, "target"].to_numpy(dtype="int64")
    return di, ti, y

ar = np.arange(SEQ) - (SEQ - 1)

def windows(di, ti, sl):
    """(B, SEQ, N_CH) window batch ending at each sample date."""
    return Cn[di[sl, None] + ar[None, :], ti[sl, None], :]

# ---------------------------------------------------------------- model
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

torch.manual_seed(SEED)
np.random.seed(SEED)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("device:", device)

class SeqNet(nn.Module):
    def __init__(self, n_ch=N_CH, hidden=96):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, num_layers=2, batch_first=True,
                            dropout=0.25)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.25),
                                  nn.Linear(hidden, 3))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

def run_epoch(model, di, ti, y, bs, train, opt=None):
    n = len(di)
    order = np.random.permutation(n) if train else np.arange(n)
    di, ti, y = di[order], ti[order], y[order]
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, preds = 0.0, []
    model.train(train)
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        xb = torch.from_numpy(windows(di, ti, sl)).to(device)
        yb = torch.from_numpy(y[sl]).to(device)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot_loss += loss.item() * len(yb)
        preds.append(logits.argmax(1).cpu().numpy())
    return tot_loss / n, f1_score(y, np.concatenate(preds), average="macro")

def train_model():
    model = SeqNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=2)
    di_f, ti_f, y_f = gather(fit_mask)
    di_v, ti_v, y_v = gather(val_mask)
    bs = 4096 if use_cuda else 2048
    max_epochs = 30 if use_cuda else 12
    best_f1, best_state, patience, since = -1, None, 5, 0
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1 = run_epoch(model, di_f, ti_f, y_f, bs, True, opt)
        with torch.no_grad():
            va_loss, va_f1 = run_epoch(model, di_v, ti_v, y_v, bs, False)
        sched.step(va_f1)
        print(f"ep {ep:02d}  train loss {tr_loss:.4f} f1 {tr_f1:.4f}  |  "
              f"val loss {va_loss:.4f} f1 {va_f1:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        if va_f1 > best_f1:
            best_f1, since = va_f1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                print("early stop"); break
    model.load_state_dict(best_state)
    return model, best_f1, ep

if use_cuda:
    with gpu_lock():
        model, best_val_f1, epochs_run = train_model()
else:
    model, best_val_f1, epochs_run = train_model()

# ---------------------------------------------------------------- test 2026
di_t, ti_t, y_t = gather(te_mask)
probs = []
model.eval()
with torch.no_grad():
    for s in range(0, len(di_t), 8192):
        sl = slice(s, min(s + 8192, len(di_t)))
        xb = torch.from_numpy(windows(di_t, ti_t, sl)).to(device)
        probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
probs = np.concatenate(probs)

score_and_save("model_lstm", samp, te_mask, probs.argmax(1), probs[:, 2],
               extra={"seq_len": SEQ, "n_channels": N_CH,
                      "device": str(device), "epochs": int(epochs_run),
                      "best_val_f1": round(float(best_val_f1), 4)})
torch.save(model.state_dict(), EXP / "results" / "model_lstm.pt")
