# Bayesian Regime Detection Engine for Equity Direction Forecasting

> **CONFIDENTIALITY NOTICE**  
> **Classification:** Strictly Private & Confidential  
> **Organization:** Zetheta Algorithms Private Limited  
> **Repository:** `bayesian-regime-detection-engine`  
> **Restrictions:** This repository and all associated documentation, code, artifacts, and figures must remain private. Do not publish, redistribute, or expose any portion of this project externally or in public portfolios.

---

## Executive Summary

The **Bayesian Regime Detection Engine for Equity Direction Forecasting** is an enterprise-grade quantitative research and inference system engineered specifically for Indian equity markets. Modern financial asset returns exhibit non-stationarity, time-varying conditional volatility, sudden structural shifts, and fat-tailed distributions that render single-state linear models fragile.

This engine reformulates market analysis through a strict probabilistic paradigm:
1. **Direction over price**: Focus on categorical regime states rather than noisy point forecasts.
2. **Probability over point forecast**: Compute full time-varying posterior probability distributions over market states.
3. **Calibrated uncertainty over false precision**: Decompose total uncertainty into epistemic (model/parameter lack of knowledge) and aleatoric (inherent market noise), backed by distribution-shift-robust conformal prediction sets.
4. **Documented ensemble over a single black box**: Aggregate diverse model families (Frequentist HMM, Bayesian HMM, Regime-Switching VAR, Bayesian Deep Learning, Foundation Models) via Bayesian Model Averaging and Constrained Stacking.

---

## The 5-Regime Taxonomy

The system classifies market dynamics into five strictly defined economic and statistical regimes:

| Regime | Macro & Market Characteristics | Canonical Quantitative Signals | Portfolio Implication |
| :--- | :--- | :--- | :--- |
| **1. Risk-On** | Sustained equity rally, broad market participation, expanding liquidity, risk appetite | Nifty 50 > 200 DMA, India VIX < 14, positive cumulative FII flows, high market breadth | Aggressive equity beta, overweight mid/small caps |
| **2. Risk-Off** | Drawdown, macro deterioration, aggressive deleveraging, flight-to-safety | Nifty 50 < 200 DMA, India VIX > 22, persistent FII outflows, INR depreciation, rising gilt yields | Capital preservation, cash, short overlay, gold |
| **3. Transitional** | Boundary zone with conflicting signals, elevated ambiguity, regime inflection | Mixed breadth, VIX compression/expansion divergence, fluctuating cross-asset correlations | Neutral posture, risk reduction, tighter stops |
| **4. Late-Cycle** | Mature expansion, narrow index leadership, stretched multiples, fatigue | Large-cap divergence from breadth, yield curve flattening, slowing earnings momentum | Defensives, quality factor tilt, trimmed beta |
| **5. Post-Shock** | Volatility stabilization post-capitulation, mean-reversion opportunity | High absolute VIX turning downward, extreme negative momentum turning positive, RSI divergence | Tactical mean-reversion, structured re-entry |

*Note: Models do not arbitrarily assign numerical indices to regime names. An automated post-fit identification and validation pipeline maps latent states to structural regimes based on verified physical characteristics (mean return, realized volatility, and drawdown dynamics).*

---

## High-Level System Architecture

The engine is engineered around a **Two-Speed Inference Architecture**:

```
                                  ┌────────────────────────┐
                                  │ Indian Market Sources  │
                                  │ (NSE, RBI, MOSPI, etc.)│
                                  └───────────┬────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │ Ingestion & Snapshot   │
                                  │ Contract Verification  │
                                  └───────────┬────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │ Leakage-Free Feature   │
                                  │ Engineering Pipeline   │
                                  └───────────┬────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │   Nightly Batch Layer   │                       │  Intraday Online Layer  │
        ├─────────────────────────┤                       ├─────────────────────────┤
        │ • Frequentist HMM       │                       │ • Bootstrap Particle    │
        │ • Bayesian HMM (MCMC)   │                       │   Filter (SMC)          │
        │ • Regime-Switching VAR  │                       │ • Bayesian Online       │
        │ • Bayesian Deep Net/BNN │                       │   Changepoint Detection │
        │ • Foundation Model Head │                       │ • Streaming Posterior   │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │  Model Ensembling:      │                       │ Online / Batch          │
        │  BMA & Constrained Stack│                       │ Reconciliation Check    │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                                 ┌───────────────────────────┐
                                 │   Conformal Prediction    │
                                 │   & Calibration Layer     │
                                 │ (APS, ECE, Reliability)   │
                                 └────────────┬──────────────┘
                                              ▼
                                 ┌───────────────────────────┐
                                 │  Central Output Contract  │
                                 │    (RegimePrediction)     │
                                 └────────────┬──────────────┘
                                              ▼
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
            ┌─────────────────────────┐               ┌─────────────────────────┐
            │   Regime-Conditioned    │               │  Investment Committee   │
            │   Monte Carlo & VaR     │               │   Reporting Generator   │
            │   Allocation Backtest   │               │   & Pack Artifacts      │
            └─────────────────────────┘               └─────────────────────────┘
```

---

## Directory Organization

```
bayesian-regime-detection-engine/
├── .gitignore                     # Leakage and artifact protection
├── pyproject.toml                 # Standard packaging and tool configurations
├── requirements.txt               # Pinned Python package dependencies
├── environment.yml                # Conda cross-language (Python + R) specification
├── README.md                      # Executive overview and operational guide
├── CLAUDE.md                      # Assistant and project instructions
├── ARCHITECTURE.md                # In-depth architectural specification
├── DATA_CONTRACT.md               # Data dictionary, lineage, and schemas
├── VALIDATION_PROTOCOL.md         # Temporal validation, leakage guards & MCMC standards
├── EXPERIMENT_PROTOCOL.md         # Experiment logging, reproducibility & registry rules
├── DEVELOPMENT_PLAN.md            # Catch-up execution schedule and milestones
│
├── config/                        # Declarative experiment and engine configs
│   ├── data.yaml                  # Asset universe, tickers, frequencies, missing rules
│   ├── models.yaml                # Model architectures, priors, and hyperparams
│   ├── validation.yaml            # Chronological splits, walk-forward windows
│   ├── ensemble.yaml              # Ensembling hurdles, BMA, stacking weights
│   ├── calibration.yaml           # Conformal alphas, APS, ECE thresholds
│   ├── online.yaml                # Particle filter, BOCPD hazard rate, streaming
│   ├── backtest.yaml              # 2019-2024 allocation overlay, transaction costs
│   └── logging.yaml               # Structured logging configuration
│
├── src/                           # Core Python source library
│   ├── data/                      # Ingestion, validation, caching, contracts
│   ├── features/                  # Return, volatility, breadth, TDA, GNN features
│   ├── models/                    # HMM, Bayesian HMM, RS-VAR, BNN, deep ensembles
│   │   └── foundation/            # Unified adapter for Chronos, TimesFM, etc.
│   ├── ensemble/                  # BMA, constrained stacking, model selector
│   ├── calibration/               # Conformal prediction, APS, ECE, reliability
│   ├── online/                    # SMC Particle filter, BOCPD, reconciliation
│   ├── simulation/                # Regime-conditioned Monte Carlo, VaR/CVaR
│   ├── backtest/                  # Dynamic allocation overlay, performance metrics
│   ├── evaluation/                # Directional score, Brier score, diagnostics
│   ├── reporting/                 # Investment Committee artifact generation
│   ├── pipeline/                  # End-to-end batch and streaming pipelines
│   └── utils/                     # Seed management, logging, experiment tracking
│
├── R/                             # Dual-language R implementation
│   ├── dependencies.R             # Package installation script
│   ├── models/                    # depmixS4 HMM, MSwM regime switching
│   ├── validation/                # R validation suite
│   └── reconciliation/            # Cross-language numerical parity scripts
│
├── tests/                         # Comprehensive multi-tier test suite
│   ├── conftest.py                # Reusable test fixtures and synthetic time series
│   ├── unit/                      # Isolated unit tests for contracts and modules
│   ├── integration/               # Pipeline execution tests
│   ├── statistical/               # Invariants (probabilities sum to 1, transition stochasticity)
│   └── leakage/                   # Look-ahead bias and feature leakage automated checks
│
├── experiments/                   # Experiment registry & metadata records
│   ├── registry/                  # JSON registry tracking all experiment runs
│   ├── configs/                   # Run-specific configurations
│   └── results/                   # Tabulated run evaluations
│
├── reports/                       # Generated analysis and publication artifacts
│   ├── figures/                   # Reliability diagrams, regime ribbons, drawdown curves
│   ├── tables/                    # Performance attribution and ECE comparisons
│   └── drafts/                    # Working deliverables and Investment Committee briefs
│
└── scripts/                       # Operational and sanity check utilities
    ├── verify_environment.py      # Environment check and dependency validation
    └── run_leakage_check.py       # Standalone data leakage verification script
```

---

## Getting Started

### 1. Environment Setup

#### Option A: Conda (Recommended for Python + R Unified Stack)
```bash
conda env create -f environment.yml
conda activate bayesian-regime-engine
```

#### Option B: Virtualenv (Python Focused)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### Verification
Execute the environment verification script to validate dependencies and toolchains:
```bash
python scripts/verify_environment.py
```

---

## Core Execution Commands

### Running Quality and Test Suites
```bash
# Run complete test suite (unit, statistical, leakage)
pytest tests/ -v

# Run statistical invariant verification only
pytest tests/statistical/ -v

# Run strict temporal leakage checks
pytest tests/leakage/ -v

# Code quality and typing checks
ruff check src/ tests/
black --check src/ tests/
mypy src/
```

---

## Model Governance & Reproducibility Standard

Every quantitative run must be fully reproducible from raw config to final probability:
1. **Config Driven**: No hardcoded hyperparameters in source code.
2. **Fixed Random Seeds**: Explicit seed injection (`src.utils.reproducibility.set_seed`).
3. **Data Snapshots**: All training pipelines record exact SHA-256 hashes of input data.
4. **MCMC Convergence Audits**: Posterior samples are verified against Gelman-Rubin $\hat{R} < 1.05$ and effective sample size criteria before any downstream inclusion.

---

## Confidentiality & Submission

This project is prepared strictly for internal research and evaluation by **Zetheta Algorithms Private Limited**. The repository transfer must follow the mandated submission workflow to `ZethetaIntern`.
