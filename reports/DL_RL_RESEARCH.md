# Deep learning & reinforcement learning approaches for the direction problem

Research pass on what's out there for fixing the under/over-perform direction problem, beyond what we've already tried (LSTM on raw sequences, tree ensembles, cost-sensitive decisions). Goal: find approaches that bring in genuinely new *structure* or *objective*, not just a fancier model on the same 74 trailing price features — since this session established that re-tuning the same inputs plateaus around 55-56% direction accuracy on true extremes.

## Why this matters for what to try next

Everything we've built so far treats each ETF independently: 74 features per (date, ticker) row, fed to a model that has no notion that a fund belongs to a category with ~50-500 peers moving together. The two most promising literature directions both attack exactly that gap — one architecturally (graph neural networks that model peer relationships explicitly), one via the training objective (ranking losses that directly optimize "who beats whom" instead of independent classification). The RL literature attacks a different gap: it reframes the problem as sequential capital allocation with a risk-adjusted reward, rather than a next-5-day classification accuracy problem — which may sidestep the "coin flip" framing rather than solve it head-on.

## Deep learning approaches

**1. Graph neural networks over the peer/category structure (most directly applicable to our data).**
Our data already has a natural graph: `category` groups funds into peer sets, exactly the structure GNN stock-ranking models are built for. The reference architecture is **HIST** (*"HIST: A Graph-based Framework for Stock Trend Forecasting"*, [arXiv:2110.13716](https://arxiv.org/pdf/2110.13716)): a temporal encoder (LSTM/GRU) produces a per-stock embedding, then a graph convolution step propagates information across "concept" edges (sector, peer-group, or other relation) before the final prediction. The core idea — don't predict each fund in isolation, let a fund's embedding be influenced by what its peers are doing that same day — maps onto our `category` column directly, with graph edges = same-category membership (or a learned/correlation-based edge weight instead of a hard binary edge). Related work: relational stock ranking / multi-relational graph attention networks (GCN/GAT over industry + price-correlation graphs) — see the broader survey in [GNN4Fintech](https://github.com/jwwthu/GNN4Fintech). This is the single idea most likely to add real information our current row-independent models structurally cannot use.

**2. Ranking losses instead of classification loss (directly reframes the metric we've been fighting).**
["On Evaluating Loss Functions for Stock Ranking"](https://arxiv.org/html/2510.14156) (2025) benchmarks pointwise, pairwise (e.g. RankNet-style), and listwise (e.g. ListMLE/LambdaRank-style) losses on a Transformer for exactly this task: daily cross-sectional stock ranking for portfolio selection. This is worth trying regardless of architecture — instead of training XGBoost/LSTM to classify under/neutral/over (a pointwise, independent-row objective, which is what produces the flip-coin confusion), train directly on **pairwise comparisons within each day**: "did fund A outperform fund B" for many (A, B) pairs on the same date. This is a different loss, not a different feature set, and could be dropped into our existing XGBoost pipeline via `objective="rank:pairwise"` (XGBoost supports this natively) as a cheap first experiment before building anything from scratch.

**3. Contrastive/self-supervised pretraining** — lower priority, but worth flagging: pretrain a sequence encoder to distinguish "same regime" vs "different regime" fund-pairs (self-supervised, no labels needed) before fine-tuning on the direction task, to squeeze more out of the price history than a purely supervised LSTM can. Less mature in this domain than GNNs/ranking losses; would be a bigger lift for uncertain payoff.

## Reinforcement learning approaches

**1. RL for portfolio allocation (PPO-based).** ["Attention-Enhanced Reinforcement Learning for Dynamic Portfolio Optimization"](https://arxiv.org/pdf/2510.06466) (2025) uses PPO with a temporal attention mechanism over price history; state = market features across assets, action = continuous portfolio weights, reward = returns net of transaction costs. Reported Sharpe 0.73 vs 0.66 for buy-and-hold in their backtest. The appeal for us: this reframes the objective away from "classify direction correctly" toward "maximize risk-adjusted portfolio return," which is arguably closer to what actually matters and sidesteps the flip-rate framing (the agent is scored on cumulative reward, not per-row accuracy).

**2. RL for relative-value / pairs trading (closest conceptual match to our peer-relative label).** Our whole problem is inherently a relative-value question — "will fund X outperform its category" is structurally the same as a pairs-trading signal (X vs. its peer basket). ["Select and Trade: Towards Unified Pair Trading with Hierarchical Reinforcement Learning"](https://arxiv.org/pdf/2301.10724) and ["Mastering Pair Trading with Risk-Aware Recurrent Reinforcement Learning"](https://arxiv.org/pdf/2304.00364) both build RL agents that select *and* size relative-value positions rather than just classify direction. This is conceptually the best-matched RL framing to our actual label construction (leave-one-out peer-relative excess), more so than generic portfolio-optimization RL.

## What I'd actually try first, given our setup

Given a single 6GB GPU and the existing 74-feature pipeline, in order of effort-to-payoff:

1. **Cheapest, fastest signal on whether "new objective" helps at all**: swap XGBoost's `objective="multi:softprob"` for `objective="rank:pairwise"` (or `rank:ndcg`) on the existing feature matrix — this is a one-line change to `model_xgboost.py`, no new architecture, and directly tests idea #2 above.
2. **Medium lift, likely biggest structural upgrade**: a small GNN (2-3 GCN/GAT layers) using `category` as the graph, with our existing 74 features as node features instead of raw price — reuses everything already built, just adds a message-passing layer before the classification/ranking head. PyTorch Geometric would be the natural library; fits comfortably on the 4050.
3. **Bigger lift, different payoff profile**: an RL pairs-trading agent scored on portfolio Sharpe rather than classification accuracy — most likely to produce a genuinely different-looking result (a backtest curve instead of a confusion matrix), but is a much larger build (needs an environment, reward shaping, and its own validation methodology) than 1 or 2.

## Honest caveat

None of this repeals the information ceiling found this session. A GNN or RL agent trained on the *same* price/volume history is still bounded by how much direction signal actually exists in that data — the 55-56% ceiling was a property of the *information*, not obviously a property of the specific models tried. GNNs and ranking losses have the best shot at moving it because they add genuinely new information (peer structure, relative comparisons) rather than just more model capacity on the same independent-row inputs. RL mainly changes what's being optimized, not what's known — useful if risk-adjusted portfolio return is really the goal, but it won't make direction calls individually less coin-flippy.

## Sources
- [On Evaluating Loss Functions for Stock Ranking: An Empirical Analysis With Transformer Model](https://arxiv.org/html/2510.14156)
- [HIST: A Graph-based Framework for Stock Trend Forecasting](https://arxiv.org/pdf/2110.13716)
- [GNN4Fintech (survey/collection of GNN finance papers)](https://github.com/jwwthu/GNN4Fintech)
- [Attention-Enhanced Reinforcement Learning for Dynamic Portfolio Optimization](https://arxiv.org/pdf/2510.06466)
- [Select and Trade: Towards Unified Pair Trading with Hierarchical Reinforcement Learning](https://arxiv.org/pdf/2301.10724)
- [Mastering Pair Trading with Risk-Aware Recurrent Reinforcement Learning](https://arxiv.org/pdf/2304.00364)
- [Reinforcement Learning Pair Trading: A Dynamic Scaling Approach](https://arxiv.org/pdf/2407.16103)
- [ACT: Anti-Crosstalk Learning for Cross-Sectional Stock Ranking](https://arxiv.org/pdf/2604.20204)
