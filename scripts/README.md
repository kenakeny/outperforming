# Utility commands

The project keeps its primary runtime entry points in the repository root:

- `etl.py` builds the main data pipeline.
- `train_models.py` trains the served five-day model family.
- `inference.py` owns the model-serving contract.
- `saudi.py` runs the Saudi transfer-learning evaluation.

Supporting commands are grouped here by responsibility. Run them from the
repository root with Python's module syntax so imports and project-relative
paths remain stable:

```bash
python -m scripts.data.build_dataset --help
python -m scripts.training.ablate_xgboost --help
python -m scripts.evaluation.backtest_vs_spy --help
python -m scripts.automation.run_notebooks --help
```

## Folders

| Folder | Contents |
| --- | --- |
| `data/` | Standalone dataset builders, support/resistance features, news ingestion, and sentiment scoring |
| `training/` | Alternative horizons, binary formulations, tuning, ablation, and standalone blends |
| `evaluation/` | Backtests, benchmark plots, execution analysis, and overfit auditing |
| `automation/` | Multi-model and notebook workflow runners |
