# Experiment Protocol & Registry Standard

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited  
> **Project:** Bayesian Regime Detection Engine for Equity Direction Forecasting

---

## 1. Reproducibility & Tracking Mandate

In quantitative research, untracked experiments and selective reporting lead to overfitting and irreproducible alpha. 

**Zero-Untracked-Work Rule**:
No experiment, model tuning run, or backtest may exist solely within an untracked scratch notebook. Every model training run must write a structured, immutable record to the **Experiment Registry** (`experiments/registry/`).

---

## 2. Experiment Record Schema

Every experiment execution generates a JSON record adhering to this schema:

```json
{
  "experiment_id": "exp_20260927_bhmm_001",
  "timestamp": "2026-09-27T15:30:00Z",
  "git_commit": "8f3b21a9c4e51082...",
  "git_branch": "feature/bayesian-hmm",
  "data_snapshot_id": "data_v1.0_2009_2024_d41d8c",
  "feature_version": "feat_v1.2_macro_vol_e2b9c0",
  "model_family": "BayesianHMM",
  "model_version": "1.0.0",
  "training_period": {
    "start": "2009-01-01",
    "end": "2017-12-31"
  },
  "calibration_period": {
    "start": "2018-01-01",
    "end": "2018-12-31"
  },
  "test_period": {
    "start": "2019-01-01",
    "end": "2024-12-31"
  },
  "random_seed": 42,
  "hyperparameters": {
    "n_states": 5,
    "covariance_type": "full",
    "mcmc_samples": 2000,
    "mcmc_tune": 1000,
    "target_accept": 0.95
  },
  "priors": {
    "dirichlet_alpha_diag": 10.0,
    "dirichlet_alpha_offdiag": 1.0,
    "mu_prior_sigma": 2.0
  },
  "metrics": {
    "out_of_sample_log_loss": 0.842,
    "brier_score": 0.215,
    "expected_calibration_error": 0.048,
    "directional_accuracy": 0.642,
    "sharpe_ratio_overlay": 1.38,
    "information_ratio": 0.94
  },
  "diagnostics": {
    "max_r_hat": 1.018,
    "min_ess_bulk": 682,
    "divergences": 0,
    "energy_bfmi": 0.62
  },
  "calibration_results": {
    "aps_marginal_coverage_90": 0.897,
    "aci_average_set_size": 1.62
  },
  "output_artifacts": {
    "posterior_trace_uri": "artifacts/traces/exp_bhmm_001.nc",
    "weights_uri": "artifacts/weights/exp_bhmm_001.pt",
    "predictions_uri": "artifacts/predictions/exp_bhmm_001.parquet"
  }
}
```

---

## 3. Storage & Integration Architecture

1. **Lightweight Internal Registry**:
   - The primary registry consists of individual JSON files stored in `experiments/registry/<experiment_id>.json`.
   - An index catalog `experiments/registry/index.json` aggregates summaries for fast querying, filtering, and model comparison.
2. **Optional MLflow Integration**:
   - The registry module (`src.utils.experiment_tracker`) provides a pluggable MLflow client that automatically logs hyperparameters, metrics, and artifact URIs if an MLflow tracking server is active.
   - If MLflow is unavailable, local file-based tracking functions autonomously with zero external dependency.

---

## 4. Multi-Model Comparison Standards

When comparing candidate models for ensemble inclusion or promotion to production:

1. **Information Criteria**:
   - For Bayesian models, compare using **WAIC** (Watanabe-Akaike Information Criterion) and **PSIS-LOO** (Pareto-Smoothed Importance Sampling Cross-Validation) via ArviZ:
     $$\Delta \text{ELPD}_{\text{LOO}} = \text{ELPD}_{\text{model 1}} - \text{ELPD}_{\text{model 2}}$$
2. **Calibration Dominance**:
   - Models with lower Expected Calibration Error (ECE) and lower Brier score on the calibration split are strictly favored over models with higher raw accuracy but poor calibration.
3. **Downstream Economic Value**:
   - Evaluate regime-overlay Information Ratio (IR) and Maximum Drawdown (MDD) reduction relative to the Nifty 50 buy-and-hold benchmark.
