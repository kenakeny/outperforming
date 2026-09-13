"""model_catgnn — peer-aware neural net: message passing over the category graph.

The structural hypothesis from reports/DL_RL_RESEARCH.md: every model so far
scores each fund as an isolated row, but the label is defined RELATIVE to
category peers -- the model should get to see what a fund's peers look like
today before calling direction. HIST-style GNN, implemented in pure torch:
for a same-category clique graph, GCN mean aggregation == masked
category-mean pooling and GAT == attention-weighted category pooling, so
message passing is done with scatter ops instead of explicit edge lists
(a 500-fund category would need 250k edges in edge-list form; this is
equivalent and far faster -- deliberate deviation from the torch_geometric
route in the plan).

Architecture (per trading day, all funds of that day as one graph):
  z0 = MLP_in(x)                                  x = 74 std. features
  z1 = LayerNorm(z0 + MLP1([z0, attnpool_cat(z0)]))    message pass 1
  z2 = LayerNorm(z1 + MLP2([z1, attnpool_cat(z1)]))    message pass 2
  score = MLP_head([z2, x])                        residual to raw features

Loss: pairwise margin ranking on same-category same-day pairs (the Step-2
objective, now with peer structure). Early stopping on val rank IC.
2-week horizon, 1y rolling window, 2026 test, fixed eval protocol.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from exp_harness import load_sweep_dataset, load_universe, gpu_lock, EXP
from eval_protocol import evaluate

HORIZON = 10
SEED = 42
HID = 128
EPOCHS, PATIENCE = 40, 8
PAIRS_PER_NODE = 8
MARGIN = 0.1

d, features, fit_m, val_m, te_m, _ = load_sweep_dataset(horizon=HORIZON,
                                                        window_years=1)
print(f"dataset: fit={fit_m.sum()} val={val_m.sum()} test={te_m.sum()}",
      flush=True)

# ---------------------------------------------------------------- features
med = d.loc[fit_m, features].median()
mu = d.loc[fit_m, features].mean()
sd = d.loc[fit_m, features].std().replace(0, 1.0)
Xall = ((d[features].fillna(med) - mu) / sd).clip(-5, 5).fillna(0.0) \
    .to_numpy(dtype="float32")

_, cat = load_universe()
cat_codes = pd.Series(pd.Categorical(
    d.index.get_level_values("ticker").map(cat)).codes,
    index=d.index).to_numpy()
rel_all = d["rel"].to_numpy(dtype="float32")
dates_all = d.index.get_level_values("date")

def day_slices(mask):
    """[(row_indices, cat_ids)] one entry per trading day inside mask."""
    idx = np.flatnonzero(mask)
    df = pd.DataFrame({"i": idx}, index=dates_all[idx])
    return [(g["i"].to_numpy(), cat_codes[g["i"].to_numpy()])
            for _, g in df.groupby(level=0)]

days_fit, days_val, days_te = (day_slices(m) for m in (fit_m, val_m, te_m))
print(f"days: fit={len(days_fit)} val={len(days_val)} test={len(days_te)}",
      flush=True)

# ---------------------------------------------------------------- model
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NF = len(features)


def attnpool_cat(h, cid, attn_w):
    """Attention-weighted mean of h within each category (message passing
    over the same-category clique, done with scatter ops)."""
    e = (h * attn_w).sum(-1)                              # (N,)
    emax = torch.full((int(cid.max()) + 1,), -torch.inf, device=h.device)
    emax = emax.scatter_reduce(0, cid, e, reduce="amax")
    a = torch.exp(e - emax[cid])
    denom = torch.zeros_like(emax).scatter_add(0, cid, a)
    a = (a / denom[cid]).unsqueeze(-1)                    # (N,1) softmax in cat
    pooled = torch.zeros(int(cid.max()) + 1, h.shape[1], device=h.device)
    pooled = pooled.index_add(0, cid, h * a)
    return pooled[cid]


class CatGNN(nn.Module):
    def __init__(self, nf=NF, hid=HID):
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
        z = self.n1(z + self.mp1(torch.cat([z, attnpool_cat(z, cid, self.attn1)], -1)))
        z = self.n2(z + self.mp2(torch.cat([z, attnpool_cat(z, cid, self.attn2)], -1)))
        return self.head(torch.cat([z, x], -1)).squeeze(-1)


def sample_pairs(cid, rel, k=PAIRS_PER_NODE, rng=None):
    """Random same-category (i, j) pairs with distinct rel, as index arrays."""
    n = len(cid)
    i = rng.integers(0, n, size=k * n)
    j = rng.integers(0, n, size=k * n)
    ok = (cid[i] == cid[j]) & (rel[i] != rel[j]) & (i != j)
    return i[ok], j[ok]


def day_scores(model, days):
    model.eval()
    out = np.empty(sum(len(ix) for ix, _ in days), dtype="float32")
    pos = 0
    with torch.no_grad():
        for ix, cc in days:
            x = torch.from_numpy(Xall[ix]).to(device)
            cid = torch.from_numpy(cc.astype(np.int64)).to(device)
            s = model(x, cid).cpu().numpy()
            out[pos:pos + len(ix)] = s
            pos += len(ix)
    return out


def rank_ic(days, scores):
    pos, ics = 0, []
    for ix, _ in days:
        s = scores[pos:pos + len(ix)]; pos += len(ix)
        ics.append(pd.Series(s).corr(pd.Series(rel_all[ix]), method="spearman"))
    return float(np.nanmean(ics))


def train():
    model = CatGNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    best_ic, best_state, since, ep_done = -np.inf, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(days_fit))
        tot, nb = 0.0, 0
        for di in order:
            ix, cc = days_fit[di]
            if len(ix) < 10:
                continue
            x = torch.from_numpy(Xall[ix]).to(device)
            cid = torch.from_numpy(cc.astype(np.int64)).to(device)
            s = model(x, cid)
            pi, pj = sample_pairs(cc, rel_all[ix], rng=rng)
            if len(pi) == 0:
                continue
            sign = torch.from_numpy(
                np.sign(rel_all[ix][pi] - rel_all[ix][pj]).astype("float32")).to(device)
            loss = F.relu(MARGIN - sign * (s[pi] - s[pj])).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        va_ic = rank_ic(days_val, day_scores(model, days_val))
        print(f"ep {ep:02d}  loss {tot / max(nb, 1):.4f}  val_ic {va_ic:+.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        ep_done = ep
        if va_ic > best_ic:
            best_ic, since = va_ic, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                print("early stop"); break
    model.load_state_dict(best_state)
    return model, best_ic, ep_done


if device.type == "cuda":
    with gpu_lock():
        model, best_val_ic, eps = train()
else:
    model, best_val_ic, eps = train()

score_te = day_scores(model, days_te)
evaluate("catgnn_rank", d, te_m, score_te,
         extra={"horizon": HORIZON, "window_years": 1, "hidden": HID,
                "epochs": int(eps), "best_val_ic": round(best_val_ic, 4),
                "loss": "pairwise-margin same-category"})
torch.save(model.state_dict(), EXP / "results" / "model_catgnn.pt")
print("model saved -> experiments/results/model_catgnn.pt")
