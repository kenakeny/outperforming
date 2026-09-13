# Model results (v3 + S/R + news)

| Model | macro-F1 | accuracy |
|---|---|---|
| LogReg +sr | 0.3633 | 0.3725 |
| LogReg +sr+news | 0.3636 | 0.3720 |
| XGBoost +sr | 0.4162 | 0.4234 |
| XGBoost +sr+news | 0.4161 | 0.4235 |
| CatBoost +sr | 0.4122 | 0.4221 |
| CatBoost +sr+news | 0.4123 | 0.4222 |
| LSTM +sr+news | 0.4168 | 0.4187 |

Baseline: uniform random macro-F1 0.3335 (always-majority 0.1696). Label = 5-day
peer-relative tercile, purged 5 trading days; see reports/LABEL_HORIZON_BUG.md.
Run 2026-07-25 on the rebuilt dataset (3,357,600 rows).