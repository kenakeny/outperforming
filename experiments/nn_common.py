"""nn_common — shared CatGNN model / preprocessing / training loop.

Single source of truth for the peer-aware GNN used by the production signal
(same reason exp_harness.py exists: model_catgnn.py and model_blend.py each
carried a copy of this class while experimenting; production code imports
from here so the architecture can't drift).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HID = 128


def standardize_stats(d, features, fit_mask):
    """(median, mean, std) computed on fit rows only."""
    med = d.loc[fit_mask, features].median()
    mu = d.loc[fit_mask, features].mean()
    sd = d.loc[fit_mask, features].std().replace(0, 1.0)
    return med, mu, sd


def standardize(X_df, med, mu, sd):
    return ((X_df.fillna(med) - mu) / sd).clip(-5, 5).fillna(0.0) \
        .to_numpy(dtype="float32")


def attnpool_cat(h, cid, attn_w):
    """Attention-weighted mean of h within each category (message passing
    over the same-category clique, via scatter ops)."""
    e = (h * attn_w).sum(-1)
    emax = torch.full((int(cid.max()) + 1,), -torch.inf, device=h.device)
    emax = emax.scatter_reduce(0, cid, e, reduce="amax")
    a = torch.exp(e - emax[cid])
    denom = torch.zeros_like(emax).scatter_add(0, cid, a)
    a = (a / denom[cid]).unsqueeze(-1)
    pooled = torch.zeros(int(cid.max()) + 1, h.shape[1], device=h.device)
    pooled = pooled.index_add(0, cid, h * a)
    return pooled[cid]


class CatGNN(nn.Module):
    """2 rounds of attention message passing over the category graph,
    residual to the raw features, scalar bullishness score per fund."""

    def __init__(self, nf, hid=HID):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, hid), nn.GELU())
        self.attn1 = nn.Parameter(torch.randn(hid) / hid ** 0.5)
        self.mp1 = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                 nn.Dropout(0.2))
        self.n1 = nn.LayerNorm(hid)
        self.attn2 = nn.Parameter(torch.randn(hid) / hid ** 0.5)
        self.mp2 = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                 nn.Dropout(0.2))
        self.n2 = nn.LayerNorm(hid)
        self.head = nn.Sequential(nn.Linear(hid + nf, hid), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(hid, 1))

    def forward(self, x, cid):
        z = self.inp(x)
        z = self.n1(z + self.mp1(torch.cat(
            [z, attnpool_cat(z, cid, self.attn1)], -1)))
        z = self.n2(z + self.mp2(torch.cat(
            [z, attnpool_cat(z, cid, self.attn2)], -1)))
        return self.head(torch.cat([z, x], -1)).squeeze(-1)


def day_slices(mask, dates_all, cat_codes):
    idx = np.flatnonzero(mask)
    df = pd.DataFrame({"i": idx}, index=dates_all[idx])
    return [(g["i"].to_numpy(), cat_codes[g["i"].to_numpy()])
            for _, g in df.groupby(level=0)]


def day_scores(model, days, Xall, device):
    model.eval()
    out = np.empty(sum(len(ix) for ix, _ in days), dtype="float32")
    pos = 0
    with torch.no_grad():
        for ix, cc in days:
            x = torch.from_numpy(Xall[ix]).to(device)
            cid = torch.from_numpy(cc.astype(np.int64)).to(device)
            out[pos:pos + len(ix)] = model(x, cid).cpu().numpy()
            pos += len(ix)
    return out


def rank_ic(days, scores, rel_all):
    pos, ics = 0, []
    for ix, _ in days:
        s = scores[pos:pos + len(ix)]; pos += len(ix)
        ics.append(pd.Series(s).corr(pd.Series(rel_all[ix]),
                                     method="spearman"))
    return float(np.nanmean(ics))


def sample_pairs(cid, rel, k, rng):
    n = len(cid)
    i = rng.integers(0, n, size=k * n)
    j = rng.integers(0, n, size=k * n)
    ok = (cid[i] == cid[j]) & (rel[i] != rel[j]) & (i != j)
    return i[ok], j[ok]


def train_catgnn(Xall, days_fit, days_val, rel_all, device, seed=42,
                 epochs=40, patience=8, pairs_per_node=8, margin=0.1,
                 verbose=True):
    """Train with pairwise margin ranking on same-category same-day pairs,
    early stop on val rank IC. Returns (model, best_val_ic, epochs_run)."""
    import time
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CatGNN(Xall.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_ic, best_state, since, ep_done = -np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for di in rng.permutation(len(days_fit)):
            ix, cc = days_fit[di]
            if len(ix) < 10:
                continue
            x = torch.from_numpy(Xall[ix]).to(device)
            cid = torch.from_numpy(cc.astype(np.int64)).to(device)
            s = model(x, cid)
            pi, pj = sample_pairs(cc, rel_all[ix], pairs_per_node, rng)
            if len(pi) == 0:
                continue
            sign = torch.from_numpy(
                np.sign(rel_all[ix][pi] - rel_all[ix][pj])
                .astype("float32")).to(device)
            loss = F.relu(margin - sign * (s[pi] - s[pj])).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va_ic = rank_ic(days_val, day_scores(model, days_val, Xall, device),
                        rel_all)
        if verbose:
            print(f"ep {ep:02d}  val_ic {va_ic:+.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        ep_done = ep
        if va_ic > best_ic:
            best_ic, since = va_ic, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                if verbose:
                    print("early stop")
                break
    model.load_state_dict(best_state)
    return model, best_ic, ep_done
