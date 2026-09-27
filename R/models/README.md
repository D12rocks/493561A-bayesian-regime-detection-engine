# R Model Implementations

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited

This directory houses parallel statistical implementations in R to provide cross-language numerical validation:

1. `hmm_depmixs4.R`: 5-state Hidden Markov Model using `depmixS4` for Gaussian and Student-t emissions.
2. `ms_var.R`: Markov-Switching Vector Autoregression using `MSwM`.
3. `bocpd.R`: Changepoint detection using `changepoint`.

All R models ingest identical input vectors exported from `tests/fixtures/` and emit outputs matching the schema in `DATA_CONTRACT.md`.
