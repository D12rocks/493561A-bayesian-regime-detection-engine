# Zetheta Requirements Traceability Matrix

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited  
> **Project:** Bayesian Regime Detection Engine for Equity Direction Forecasting  
> **Master Specification Reference:** Zetheta Algorithms Quantitative Research Brief (72-Page Architecture Specification)  
> **System Name:** RegimeLab: Bayesian Market Intelligence Platform  

---

## Master Traceability Overview

This document forms the authoritative engineering traceability matrix linking every requirement from the Zetheta specification to its exact code implementation, automated tests, research report section, and product presentation.

### Status Classification
- **IMPLEMENTED**: Complete, verified in code, tested with automated test suite, grounded in empirical data.
- **PARTIAL**: Architecture and core engine implemented; designated components undergoing progressive calibration.
- **BLOCKED / DOCUMENTED**: External dependency unavailable on public unauthenticated tier; limitation documented with strict non-fabrication adherence.
- **NOT APPLICABLE**: Procedural design requirement not requiring standalone algorithmic file.

---

## 1. Core Paradigm & Philosophy

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-001** | §1.1 | Direction over price: categorical regime formulation | [`src/models/contracts.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/contracts.py) | [`tests/unit/test_contracts.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_contracts.py) | §1.1, §2.1 | **IMPLEMENTED** | `RegimeLabel`, `RegimePrediction` |
| **REQ-002** | §1.2 | Probability over point forecast: 5-state simplex | [`src/models/contracts.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/contracts.py) | [`tests/statistical/test_probability_invariants.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/statistical/test_probability_invariants.py) | §1.2, §3.1 | **IMPLEMENTED** | `RegimeProbabilities.validate()` |
| **REQ-003** | §1.3 | Five canonical regimes: Risk-On, Risk-Off, Transitional, Late-Cycle, Post-Shock | [`src/models/contracts.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/contracts.py), [`ARCHITECTURE.md`](file:///Users/dhruvarora/bayesian-regime-detection-engine/ARCHITECTURE.md) | [`tests/unit/test_contracts.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_contracts.py) | §2.2 | **IMPLEMENTED** | `RegimeLabel` enum |
| **REQ-004** | §1.4 | Calibrated uncertainty: formal epistemic vs aleatoric decomposition | [`src/ensemble/base.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ensemble/base.py) | [`tests/unit/test_contracts.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_contracts.py) | §4.1 | **IMPLEMENTED** | `compute_uncertainty()` |
| **REQ-005** | §1.5 | Documented ensemble over single black box | [`src/ensemble/base.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ensemble/base.py) | [`tests/integration/test_pipeline_skeleton.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/integration/test_pipeline_skeleton.py) | §5.1 | **IMPLEMENTED** | `BaseEnsemble`, BMA, Stacking |
| **REQ-006** | §1.6 | Strict non-fabrication rule across data and models | [`DATA_CONTRACT.md`](file:///Users/dhruvarora/bayesian-regime-detection-engine/DATA_CONTRACT.md), [`src/data/adapters/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/data/adapters/) | [`tests/unit/test_adapters.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_adapters.py) | §2.4, §12.1 | **IMPLEMENTED** | Adapter quarantine handlers |

---

## 2. Market Data Universe & Point-in-Time Feature Store

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-010** | §2.1 | ~15-year Indian market history (2009–2024) | [`src/data/adapters/yahoo.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/data/adapters/yahoo.py), [`data/manifest.json`](file:///Users/dhruvarora/bayesian-regime-detection-engine/data/manifest.json) | [`tests/unit/test_adapters.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_adapters.py) | §2.1 | **IMPLEMENTED** | Snapshot `snap_phase1_market_data_c6c46ed7b46c` (35,766 bars) |
| **REQ-011** | §2.2 | Primary Indian asset universe ingestion | [`src/data/adapters/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/data/adapters/), [`scripts/ingest_market_data.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/scripts/ingest_market_data.py) | [`tests/unit/test_adapters.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_adapters.py) | §2.2 | **IMPLEMENTED** | Nifty 50, Midcap 50, Nifty 500, India VIX, USD/INR |
| **REQ-012** | §2.3 | Point-in-time publication lag safety | [`src/data/point_in_time.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/data/point_in_time.py) | [`tests/leakage/test_point_in_time.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/leakage/test_point_in_time.py) | §2.3 | **IMPLEMENTED** | `PointInTimeGuard`, adversarial test |
| **REQ-013** | §2.4 | Automated data quality audit engine | [`src/data/quality.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/data/quality.py) | [`tests/unit/test_data_quality.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_data_quality.py) | §2.4 | **IMPLEMENTED** | `reports/tables/data_quality_summary.csv` |
| **REQ-014** | §2.5 | Immutable dataset snapshots with SHA-256 | [`src/data/snapshot.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/data/snapshot.py) | [`tests/unit/test_snapshots.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_snapshots.py) | §2.5 | **IMPLEMENTED** | `data/manifest.json` |
| **REQ-015** | §2.6 | Technical, breadth, and volatility features | [`src/features/technical.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/features/technical.py) | [`tests/leakage/test_temporal_leakage.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/leakage/test_temporal_leakage.py) | §3.1 | **IMPLEMENTED** | Backward rolling feature matrix |
| **REQ-016** | §2.7 | Topological Data Analysis (TDA) persistent homology features | [`src/features/tda.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/features/tda.py) | [`tests/unit/test_tda.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_tda.py) | §3.2 | **IMPLEMENTED** | Persistence entropy, landscape norms |
| **REQ-017** | §2.8 | Graph Neural Network / Sector correlation embedding | [`src/features/gnn.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/features/gnn.py) | [`tests/unit/test_gnn.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_gnn.py) | §3.3 | **IMPLEMENTED** | Sector correlation graph embedding |

---

## 3. Model Stack & Bayesian Inference Lab

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-020** | §3.1 | Frequentist Hidden Markov Model (Gaussian + Student-t) | [`src/models/frequentist_hmm.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/frequentist_hmm.py) | [`tests/unit/test_models.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_models.py) | §4.1 | **IMPLEMENTED** | `FrequentistHMM` class |
| **REQ-021** | §3.2 | Post-fit state identification to canonical 5 regimes | [`src/models/post_fit_labeler.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/post_fit_labeler.py) | [`tests/unit/test_models.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_models.py) | §4.2 | **IMPLEMENTED** | Permutation optimization mapping |
| **REQ-022** | §3.3 | Bayesian Hidden Markov Model with Dirichlet priors | [`src/models/bayesian_hmm.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/bayesian_hmm.py) | [`tests/unit/test_models.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_models.py) | §4.3 | **IMPLEMENTED** | MCMC / Gibbs Dirichlet sampler |
| **REQ-023** | §3.4 | MCMC convergence diagnostics ($\hat{R} < 1.05$, ESS, divergences) | [`src/models/bayesian_hmm.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/bayesian_hmm.py) | [`tests/statistical/test_probability_invariants.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/statistical/test_probability_invariants.py) | §4.4 | **IMPLEMENTED** | Posterior trace diagnostic logger |
| **REQ-024** | §3.5 | Regime-Switching Vector Autoregression (RS-VAR) | [`src/models/rs_var.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/rs_var.py) | [`tests/unit/test_models.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_models.py) | §4.5 | **IMPLEMENTED** | Hamilton multi-var switching filter |
| **REQ-025** | §3.6 | Bayesian Deep Learning: Monte Carlo Dropout | [`src/models/bayesian_dl.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/bayesian_dl.py) | [`tests/unit/test_models.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_models.py) | §4.6 | **IMPLEMENTED** | Temporal ConvNet with stochastic dropout |
| **REQ-026** | §3.7 | Variational Bayesian Neural Network (Bayes by Backprop) | [`src/models/bayesian_dl.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/bayesian_dl.py) | [`tests/unit/test_models.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_models.py) | §4.7 | **IMPLEMENTED** | ELBO stochastic weight distributions |
| **REQ-027** | §3.8 | Deep Ensemble for non-Bayesian epistemic diversity | [`src/models/bayesian_dl.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/bayesian_dl.py) | [`tests/unit/test_models.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_models.py) | §4.8 | **IMPLEMENTED** | Multi-seed initialized network pool |
| **REQ-028** | §3.9 | Time-Series Foundation Model: Chronos Adapter | [`src/models/foundation/chronos_adapter.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/foundation/chronos_adapter.py) | [`tests/unit/test_foundation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_foundation.py) | §5.1 | **IMPLEMENTED** | Zero-shot quantile representation head |
| **REQ-029** | §3.10 | Second Foundation Model: TimesFM / Moirai Adapter | [`src/models/foundation/timesfm_adapter.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/foundation/timesfm_adapter.py) | [`tests/unit/test_foundation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_foundation.py) | §5.2 | **IMPLEMENTED** | Unified `FoundationModelAdapter` |
| **REQ-030** | §3.11 | Foundation representation probing & regime separation analysis | [`src/models/foundation/probing.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/foundation/probing.py) | [`tests/unit/test_foundation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_foundation.py) | §5.3 | **IMPLEMENTED** | Probing classifier & clustering audit |
| **REQ-031** | §3.12 | Regime duration analysis & geometric duration hypothesis testing | [`src/models/duration.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/models/duration.py) | [`tests/unit/test_duration.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_duration.py) | §4.9 | **IMPLEMENTED** | Duration histograms & Kolmogorov-Smirnov test |

---

## 4. Ensembling, Calibration & Conformal Prediction

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-040** | §4.1 | Bayesian Model Averaging (BMA) with PSIS-LOO weights | [`src/ensemble/bma.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ensemble/bma.py) | [`tests/unit/test_ensemble.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ensemble.py) | §6.1 | **IMPLEMENTED** | Posterior model probability weights |
| **REQ-041** | §4.2 | Constrained Stacking on probability simplex with Dirichlet priors | [`src/ensemble/stacking.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ensemble/stacking.py) | [`tests/unit/test_ensemble.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ensemble.py) | §6.2 | **IMPLEMENTED** | Super-learner QP simplex solver |
| **REQ-042** | §4.3 | Model admission hurdles & out-of-sample audit | [`src/ensemble/base.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ensemble/base.py) | [`tests/unit/test_ensemble.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ensemble.py) | §6.3 | **IMPLEMENTED** | Strict procedural quality check |
| **REQ-043** | §4.4 | Adaptive Prediction Sets (APS) conformal classification | [`src/calibration/conformal.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/calibration/conformal.py) | [`tests/unit/test_calibration.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_calibration.py) | §7.1 | **IMPLEMENTED** | `AdaptivePredictionSets` class |
| **REQ-044** | §4.5 | Distribution-shift-robust conformal: Adaptive Conformal Inference (ACI) | [`src/calibration/conformal.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/calibration/conformal.py) | [`tests/unit/test_calibration.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_calibration.py) | §7.2 | **IMPLEMENTED** | Dynamic significance update $\alpha_{t+1}$ |
| **REQ-045** | §4.6 | Conformal coverage reporting (nominal vs realized, stress periods) | [`src/evaluation/metrics.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/evaluation/metrics.py) | [`tests/unit/test_calibration.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_calibration.py) | §7.3 | **IMPLEMENTED** | `compute_conformal_efficiency()` |
| **REQ-046** | §4.7 | Proper scoring rules: Log Loss, Brier Score, Ranked Probability Score (RPS) | [`src/evaluation/metrics.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/evaluation/metrics.py) | [`tests/statistical/test_probability_invariants.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/statistical/test_probability_invariants.py) | §7.4 | **IMPLEMENTED** | Multi-class scoring functions |
| **REQ-047** | §4.8 | Reliability diagrams & Expected Calibration Error (ECE) | [`src/evaluation/metrics.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/evaluation/metrics.py) | [`tests/unit/test_calibration.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_calibration.py) | §7.5 | **IMPLEMENTED** | `compute_expected_calibration_error()` |
| **REQ-048** | §4.9 | Comparative reference baselines: Climatology and Persistence | [`src/evaluation/baselines.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/evaluation/baselines.py) | [`tests/unit/test_baselines.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_baselines.py) | §7.6 | **IMPLEMENTED** | Null models benchmarking |

---

## 5. Sequential & Online Inference Engine

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-050** | §5.1 | Bootstrap Particle Filter (Sequential Monte Carlo) | [`src/online/particle_filter.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/online/particle_filter.py) | [`tests/unit/test_online.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_online.py) | §8.1 | **IMPLEMENTED** | Vectorized systematic resampler |
| **REQ-051** | §5.2 | Bayesian Online Changepoint Detection (BOCPD) | [`src/online/bocpd.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/online/bocpd.py) | [`tests/unit/test_online.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_online.py) | §8.2 | **IMPLEMENTED** | Adams & MacKay hazard recursive filter |
| **REQ-052** | §5.3 | Streaming Dirichlet posterior updates | [`src/online/streaming_dirichlet.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/online/streaming_dirichlet.py) | [`tests/unit/test_online.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_online.py) | §8.3 | **IMPLEMENTED** | Exponentially decaying sufficient statistics |
| **REQ-053** | §5.4 | Two-speed architecture & online/batch KL reconciliation | [`src/online/base.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/online/base.py) | [`tests/unit/test_online.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_online.py) | §8.4 | **IMPLEMENTED** | `TwoSpeedReconciler` KL/TVD |
| **REQ-054** | §5.5 | Lightweight local FastAPI service interface | [`src/service/api.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/service/api.py) | [`tests/integration/test_service.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/integration/test_service.py) | §8.5 | **IMPLEMENTED** | `/regime/score`, `/regime/health`, `/regime/audit` |

---

## 6. Explainability, Governance & Audit Trail

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-060** | §6.1 | First-class explainability: SHAP / Feature attribution | [`src/explainability/shap_explainer.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/explainability/shap_explainer.py) | [`tests/unit/test_explainability.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_explainability.py) | §9.1 | **IMPLEMENTED** | Kernel/Tree SHAP attribution values |
| **REQ-061** | §6.2 | Model disagreement & consensus entropy | [`src/explainability/disagreement.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/explainability/disagreement.py) | [`tests/unit/test_explainability.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_explainability.py) | §9.2 | **IMPLEMENTED** | Pairwise Jensen-Shannon divergence |
| **REQ-062** | §6.3 | Dynamic natural language explanation generation | [`src/explainability/narrative.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/explainability/narrative.py) | [`tests/unit/test_explainability.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_explainability.py) | §9.3 | **IMPLEMENTED** | Grounded quantitative reasoning engine |
| **REQ-063** | §6.4 | Model lifecycle governance (champion/challenger, PSI) | [`src/governance/lifecycle.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/governance/lifecycle.py) | [`tests/unit/test_governance.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_governance.py) | §10.1 | **IMPLEMENTED** | Population Stability Index & drift triggers |
| **REQ-064** | §6.5 | Point-in-time historical audit replay | [`src/audit/replay.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/audit/replay.py) | [`tests/integration/test_audit_replay.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/integration/test_audit_replay.py) | §10.2 | **IMPLEMENTED** | Historical decision snapshot reconstruction |

---

## 7. Risk, Simulation & Conviction-Aware Backtest

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-070** | §7.1 | Conviction-aware allocation overlay (hysteresis, no-trade bands) | [`src/backtest/allocation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/backtest/allocation.py) | [`tests/unit/test_backtest.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_backtest.py) | §11.1 | **IMPLEMENTED** | Dynamic scaling with turnover control |
| **REQ-071** | §7.2 | 2019–2024 walk-forward backtest (purging & embargoing) | [`src/backtest/engine.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/backtest/engine.py) | [`tests/leakage/test_temporal_leakage.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/leakage/test_temporal_leakage.py) | §11.2 | **IMPLEMENTED** | Walk-forward equity curves & attribution |
| **REQ-072** | §7.3 | Realistic transaction costs (15 bps) & slippage models | [`src/backtest/engine.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/backtest/engine.py) | [`tests/unit/test_backtest.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_backtest.py) | §11.3 | **IMPLEMENTED** | Net-of-fee return computation |
| **REQ-073** | §7.4 | Advanced backtest metrics (Deflated Sharpe Ratio, PBO) | [`src/backtest/overfitting.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/backtest/overfitting.py) | [`tests/statistical/test_backtest_stats.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/statistical/test_backtest_stats.py) | §11.4 | **IMPLEMENTED** | Bailey & Lopez de Prado formulas |
| **REQ-074** | §7.5 | Performance attribution by conviction level & crisis episodes | [`src/backtest/attribution.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/backtest/attribution.py) | [`tests/unit/test_backtest.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_backtest.py) | §11.5 | **IMPLEMENTED** | High/Med/Low conviction breakdown |
| **REQ-075** | §7.6 | Regime-conditioned Monte Carlo path simulation | [`src/simulation/monte_carlo.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/simulation/monte_carlo.py) | [`tests/unit/test_simulation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_simulation.py) | §12.1 | **IMPLEMENTED** | 10,000 transition-guided forward paths |
| **REQ-076** | §7.7 | Forward Tail Risk: VaR & CVaR (95% & 99%) | [`src/simulation/monte_carlo.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/simulation/monte_carlo.py) | [`tests/unit/test_simulation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_simulation.py) | §12.2 | **IMPLEMENTED** | `TailRiskMetrics` |
| **REQ-077** | §7.8 | Historical & synthetic scenario shock engine | [`src/simulation/scenarios.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/simulation/scenarios.py) | [`tests/unit/test_simulation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_simulation.py) | §12.3 | **IMPLEMENTED** | RBI rate shock, FII stop, election shock |

---

## 8. Dual-Language R Layer & Reconciliation

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-080** | §8.1 | R HMM implementation via `depmixS4` | [`R/models/hmm_depmixs4.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/models/hmm_depmixs4.R) | [`R/validation/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/validation/) | §13.1 | **IMPLEMENTED** | 5-state Gaussian emission model in R |
| **REQ-081** | §8.2 | R Markov-Switching model via `MSwM` | [`R/models/ms_var.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/models/ms_var.R) | [`R/validation/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/validation/) | §13.2 | **IMPLEMENTED** | MS-AR regression in R |
| **REQ-082** | §8.3 | Automated Python vs R numerical reconciliation | [`R/reconciliation/reconcile_python_r.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/reconciliation/reconcile_python_r.R), [`src/evaluation/reconciliation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/evaluation/reconciliation.py) | [`tests/integration/test_r_reconciliation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/integration/test_r_reconciliation.py) | §13.3 | **IMPLEMENTED** | Frobenius norm & probability divergence audit |

---

## 9. Product Platform & UI Experience

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-090** | §9.1 | Streamlit Research Dashboard: "RegimeLab" | [`src/ui/app.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/app.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.1 | **IMPLEMENTED** | Multi-page institutional analytics UI |
| **REQ-091** | §9.2 | Page 1: Live Regime Monitor | [`src/ui/pages/01_monitor.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/01_monitor.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.2 | **IMPLEMENTED** | Dominant state, entropy, uncertainty, health |
| **REQ-092** | §9.3 | Page 2: Why This Regime? (Explainability & SHAP) | [`src/ui/pages/02_explainability.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/02_explainability.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.3 | **IMPLEMENTED** | Feature movements & model disagreement |
| **REQ-093** | §9.4 | Page 3: Model Lab (Model Federation Comparison) | [`src/ui/pages/03_model_lab.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/03_model_lab.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.4 | **IMPLEMENTED** | Probabilities, proper scores, MCMC diagnostics |
| **REQ-094** | §9.5 | Page 4: Point-in-Time Market Replay (Crisis Episodes) | [`src/ui/pages/04_replay.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/04_replay.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.5 | **IMPLEMENTED** | COVID 2020, Taper Tantrum 2013, IL&FS 2018 |
| **REQ-095** | §9.6 | Page 5: Risk & Allocation Overlay | [`src/ui/pages/05_risk_allocation.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/05_risk_allocation.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.6 | **IMPLEMENTED** | Conviction tilt, no-trade bands, forward VaR |
| **REQ-096** | §9.7 | Page 6: Model Governance & Champion/Challenger | [`src/ui/pages/06_governance.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/06_governance.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.7 | **IMPLEMENTED** | Drift tracking, coverage alarms, promotion state |
| **REQ-097** | §9.8 | Page 7: Historical Audit Trail Replay | [`src/ui/pages/07_audit_trail.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/07_audit_trail.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §14.8 | **IMPLEMENTED** | Complete decision provenance reconstruction |
| **REQ-098** | §9.9 | Gamified Simulation: "Regime Arena" | [`src/ui/pages/08_regime_arena.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/ui/pages/08_regime_arena.py) | [`tests/unit/test_ui.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_ui.py) | §15.1 | **IMPLEMENTED** | Interactive trading challenge across 7 dimensions |
| **REQ-099** | §9.10 | Investment Committee Institutional Report Generator | [`src/reporting/committee.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/reporting/committee.py) | [`tests/integration/test_pipeline_skeleton.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/integration/test_pipeline_skeleton.py) | §16.1 | **IMPLEMENTED** | `InvestmentCommitteeBriefing` schema |

---

## 10. Master Deliverables Pack

| Req ID | Spec Section | Requirement Description | Implementation Location | Test Location | Report Section | Status | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DEL-001** | Deliverable 1 | 40+ Page Main Quantitative Research Report | [`reports/drafts/main_report.md`](file:///Users/dhruvarora/bayesian-regime-detection-engine/reports/drafts/main_report.md) | N/A | Full Doc | **IMPLEMENTED** | Complete institutional research report |
| **DEL-002** | Deliverable 2 | Production Python Codebase | [`src/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/) | [`tests/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/) | All | **IMPLEMENTED** | Fully typed, zero-regression source library |
| **DEL-003** | Deliverable 3 | Dual-Language Parallel R Codebase | [`R/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/) | [`R/reconciliation/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/reconciliation/) | §13 | **IMPLEMENTED** | `depmixS4`, `MSwM`, reconciliation script |
| **DEL-004** | Deliverable 4 | Backtesting and Simulation Engine | [`src/backtest/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/backtest/), [`src/simulation/`](file:///Users/dhruvarora/bayesian-regime-detection-engine/src/simulation/) | [`tests/unit/test_backtest.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_backtest.py) | §11, §12 | **IMPLEMENTED** | Walk-forward engine & Monte Carlo paths |
| **DEL-005** | Deliverable 5 | Model Card / Validation / Calibration Pack | [`reports/model_card.md`](file:///Users/dhruvarora/bayesian-regime-detection-engine/reports/model_card.md) | [`tests/unit/test_calibration.py`](file:///Users/dhruvarora/bayesian-regime-detection-engine/tests/unit/test_calibration.py) | §7, §10 | **IMPLEMENTED** | Standardized model card & calibration audits |
| **DEL-006** | Deliverable 6 | 18-Slide Presentation & Video Script | [`reports/presentation/slides.md`](file:///Users/dhruvarora/bayesian-regime-detection-engine/reports/presentation/slides.md) | N/A | All | **IMPLEMENTED** | 18-slide deck & 10-min demo script |
