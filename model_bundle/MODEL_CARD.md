# Production model bundle — ETF peer-relative direction signal

Trained 2026-07-20. Predicts whether a US ETF will **outperform / underperform**
its category peers over the **next 10 trading days (~2 weeks)**, peer-relative
(leave-one-out category mean → per-day terciles).

## Files

| file | what it is | how to load |
|---|---|---|
| `stage1.ubj` | XGBoost booster — **extreme gate**: P(fund is in either 15% tail vs the neutral 70%) | `xgboost.Booster().load_model("stage1.ubj")` |
| `stage2.ubj` | XGBoost booster — **direction**: P(top tail \| extreme), tuned (depth 6) | `xgboost.Booster().load_model("stage2.ubj")` |
| `catgnn.pt` | PyTorch state_dict — **CatGNN**: peer-graph net, attention message-passing over category cliques | `CatGNN(nf, hid).load_state_dict(torch.load(...))` |
| `scaler.parquet` | per-feature median / mean / std (fit-rows only) — GNN input standardization | `pd.read_parquet` |
| `config.json` | feature list (81), category list (36), blend weight, hyperparams, metadata | `json.load` |

## Inference formula

```
s_xgb   = P_extreme * (2 * P_direction - 1)          # two-stage XGB score
s_gnn   = CatGNN(standardized_features, category_id) # peer-graph score
score   = w * zscore_daily(s_xgb) + (1-w) * zscore_daily(s_gnn)   # w = config["w_xgb"] = 0.2
```
Per trading day: rank funds by `score`. Highest = most likely to outperform peers,
lowest = underperform. Trade the top/bottom K% by conviction (|score − daily median|).

## How it was built (config.json)
- horizon 10 trading days, 1-year rolling training window, embargoed walk-forward
- 81 features: 74 base price/volume/peer families + 7 overnight/intraday (exp_15)
- blend weight w_xgb = 0.2 (chosen on validation rank IC only)
- validation blend rank IC = 0.180

## 2026 out-of-sample performance (held out, never trained on)
- direction accuracy on true extremes: 0.554
- daily rank IC: 0.086
- 90/5/5 long-short strategy: ~34% net annualized, Sharpe ~2.1 (formation-day approx)
- full daily-accounted long-short book: +11.7% / Sharpe 1.48 / −5.1% max DD (vs SPY +11.3% / 1.56 / −8.9%)

## Reproduce / retrain / predict
- retrain from raw data:  `python experiments/train_production.py`
- today's ranked calls:   `python experiments/predict_signal.py [--k 0.03]`
- self-contained walkthrough: `notebooks/06_production_model.ipynb`

The CatGNN architecture is defined in `experiments/nn_common.py` and inlined in the
notebook. `stage1/stage2` are plain XGBoost — no custom code needed to load.
