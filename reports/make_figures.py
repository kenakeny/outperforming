"""generate the eda dashboard pngs into reports/. run: python reports/make_figures.py
(needs data/raw + data/processed from notebooks 01-03). the notebooks also render every
one of these plots inline; this just bundles the key ones for slides."""
import pathlib, yaml
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style='whitegrid')

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
RAW, PROC = ROOT / 'data/raw', ROOT / 'data/processed'
REP = ROOT / 'reports'; REP.mkdir(exist_ok=True)

prices = pd.read_parquet(RAW / 'prices.parquet'); prices.index.name = 'date'; prices.columns.name = 'ticker'
md = pd.read_parquet(RAW / 'metadata.parquet')
labels = pd.read_parquet(PROC / 'labels.parquet')
# dataset.parquet is the superseded v1 panel, retired to data/_deprecated/ on
# 2026-07-26. These figures were generated from it on 2026-07-09 and are kept as
# a record; regenerating them against dataset_v3.parquet would need the column
# names below reconciled first.
dataset = pd.read_parquet(ROOT / 'data/_deprecated' / 'dataset.parquet')

bad = pd.Series(False, index=md.index)
for col in ['is_leveraged', 'bad_ticks', 'is_delisted']:
    if col in md.columns:
        bad |= md[col].fillna(False)
clean = prices.drop(columns=[c for c in md.index[bad] if c in prices.columns])
grp = md['category_group']

ORDER = ['Underperform', 'Neutral', 'Outperform']
COLORS = {'Underperform': '#d9534f', 'Neutral': '#aaaaaa', 'Outperform': '#5cb85c'}

# ---- figure 1: universe & label ----
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Fund outperformance - universe & label', fontsize=15, weight='bold')

reasons = pd.Series({'leveraged/\ninverse': int(md.get('is_leveraged', pd.Series(dtype=bool)).sum()),
                     'bad\nticks': int(md.get('bad_ticks', pd.Series(dtype=bool)).sum()),
                     'delisted/\nstale': int(md.get('is_delisted', pd.Series(dtype=bool)).sum())})
ax[0, 0].bar(reasons.index, reasons.values, color=['#c44', '#c84', '#48c'])
ax[0, 0].set_title(f'funds dropped by reason  (kept {clean.shape[1]} of {prices.shape[1]})')
for i, v in enumerate(reasons.values):
    ax[0, 0].text(i, v, str(v), ha='center', va='bottom')

vc = grp.reindex(clean.columns).value_counts()
ax[0, 1].barh(vc.index[::-1], vc.values[::-1], color='#4a7')
ax[0, 1].set_title('funds per category group')

cov = clean.notna().sum()
ax[0, 2].hist(cov, bins=40, color='#57a')
ax[0, 2].set_title('history: trading days per fund'); ax[0, 2].set_xlabel('days')

bal = labels['label'].value_counts().reindex(ORDER)
ax[1, 0].bar(ORDER, bal.values, color=[COLORS[c] for c in ORDER])
ax[1, 0].set_title(f'class balance  (thr={cfg["labels"]["threshold"]})')
for i, v in enumerate(bal.values):
    ax[1, 0].text(i, v, f'{v/bal.sum()*100:.0f}%', ha='center', va='bottom')

lab = labels['label']; tick = lab.index.get_level_values('ticker')
neu = lab.eq('Neutral').groupby(tick.map(grp)).mean().mul(100).sort_values()
ax[1, 1].barh(neu.index, neu.values, color='#a58')
ax[1, 1].set_title('% Neutral by group  (bonds hug peers, sectors swing)')
ax[1, 1].axvline(50, color='k', ls=':', lw=.8)

yr = lab.index.get_level_values('date').year
pct_neu = lab.eq('Neutral').groupby(yr).mean().mul(100)
ax[1, 2].bar(pct_neu.index.astype(str), pct_neu.values, color='#789')
ax[1, 2].set_title('% Neutral by year  (drops in volatile 2020/2022)')
ax[1, 2].tick_params(axis='x', rotation=45); ax[1, 2].axhline(50, color='k', ls=':', lw=.8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(REP / '01_universe_and_label.png', dpi=110); plt.close(fig)
print('wrote 01_universe_and_label.png')

# ---- figure 2: spread & features ----
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Fund outperformance - return spread & features', fontsize=15, weight='bold')

rel_bps = (labels['rel'].dropna() * 1e4).clip(-300, 300)
ax[0, 0].hist(rel_bps, bins=100, color='#69b')
ax[0, 0].set_title('relative fwd-5d return (bps, clipped +/-300)'); ax[0, 0].set_xlabel('bps')
ax[0, 0].axvline(0, color='k', lw=.8)

disp = labels['rel'].groupby(labels.index.get_level_values('ticker').map(grp)).std().mul(1e4).sort_values()
ax[0, 1].barh(disp.index, disp.values, color='#b76')
ax[0, 1].set_title('within-category rel spread (bps std)')

ret = clean.pct_change(fill_method=None)
gcol = grp.reindex(clean.columns)
for g in vc.head(6).index:
    cols = gcol[gcol == g].index
    comp = (1 + ret[cols].mean(axis=1).fillna(0)).cumprod()
    ax[0, 2].plot(comp.index, comp.values, lw=1, label=g)
ax[0, 2].set_title('growth of $1 - category-group composites'); ax[0, 2].legend(fontsize=7)

feat_cols = [c for c in dataset.columns if c not in ('label', 'rel', 'category')]
sns.heatmap(dataset[feat_cols].corr(), ax=ax[1, 0], cmap='coolwarm', center=0, cbar_kws={'shrink': .6})
ax[1, 0].set_title('feature correlation'); ax[1, 0].tick_params(labelsize=6)

mu = dataset.groupby('label')[feat_cols].mean()
sep = ((mu.loc['Outperform'] - mu.loc['Underperform']).abs() / dataset[feat_cols].std()).sort_values()
ax[1, 1].barh(sep.index, sep.values, color='#7a9')
ax[1, 1].set_title('feature signal: |Out-Under| / std  (all tiny!)'); ax[1, 1].tick_params(labelsize=7)

vb = dataset.groupby('label')['vol_20d'].mean().reindex(ORDER)
ax[1, 2].bar(ORDER, vb.values, color=[COLORS[c] for c in ORDER])
ax[1, 2].set_title('mean realized vol by label\n(movers deviate, neutral is calm)')
for i, v in enumerate(vb.values):
    ax[1, 2].text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(REP / '02_spread_and_features.png', dpi=110); plt.close(fig)
print('wrote 02_spread_and_features.png')
