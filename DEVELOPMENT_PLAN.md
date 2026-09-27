# Catch-Up Execution Plan & Research Roadmap

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited  
> **Project:** Bayesian Regime Detection Engine for Equity Direction Forecasting

---

## 1. Context & Objective

The authoritative project brief specifies a 15-day research sprint. Because project milestones must be accelerated while preserving absolute scientific rigor, this **Catch-Up Execution Plan** compresses the 15-day workflow into **5 High-Velocity Work Packages (Sprints)**.

**Critical Mandate**: No mandatory requirement may be dropped or simplified into a trivial toy. All 28 technical components (from Bayesian HMMs to Foundation Models, BOCPD, Adaptive Conformal Sets, and R Reconciliation) are fully scheduled with explicit validation milestones and deliverables.

---

## 2. Sprint Compression Matrix

```
ORIGINAL 15-DAY TIMELINE               COMPRESSED CATCH-UP PHASES
Day 1: Setup & Env             ──┐
Day 2: Data Ingestion          ──┼──►  PHASE 1: Foundations, Ingestion & Feature Engineering
Day 3: Feature Eng (TDA/GNN)   ──┘
─────────────────────────────────────────────────────────────────────────────────────────────
Day 4: Frequentist HMM         ──┐
Day 5: Bayesian HMM            ──┼──►  PHASE 2: Core Regime Model Stack (Econometric + BNN)
Day 6: RS-VAR                  ──┤
Day 7: Bayesian DL (MC/BNN)    ──┘
─────────────────────────────────────────────────────────────────────────────────────────────
Day 8: Foundation Models       ──┐
Day 9: Ensembling (BMA/Stack)  ──┼──►  PHASE 3: Foundation Adapter, Ensembling & Calibration
Day 11: Conformal & Calib      ──┘
─────────────────────────────────────────────────────────────────────────────────────────────
Day 10: Online SMC & BOCPD     ──┐
Day 13: Backtest & Monte Carlo ──┼──►  PHASE 4: Online Streaming, Risk Engine & Backtest
Day 12: Indian Case Studies    ──┘
─────────────────────────────────────────────────────────────────────────────────────────────
Day 14: R Layer & Polish       ──┐
Day 15: Deliverables Handover  ──┴──►  PHASE 5: Dual-Language R Parity, Reporting & Packaging
```

---

## 3. Detailed Work Packages

### Phase 1: Foundations, Ingestion & Feature Engineering (Days 1–3 Compressed)
- **Objective**: Establish rock-solid data pipeline, storage contracts, and feature library.
- **Key Deliverables**:
  - Ingestion connectors for Nifty 50, Midcap 100, Smallcap 100, India VIX, USD/INR, 10Y Gilt, FII/DII flows, SIP totals (2009–2024).
  - Feature engineering pipeline: returns, EWMA volatilities, breadth metrics, macro lags.
  - Genuine lightweight TDA (persistence entropy) and GNN sector correlation embeddings.
  - Automated leakage check suite verifying zero lookahead bias across all features.
- **Validation Gate**: Pass all `tests/leakage/` and contract tests; generate `data_snapshot_v1.0`.

### Phase 2: Core Regime Model Stack (Days 4–7 Compressed)
- **Objective**: Build and validate the primary statistical and Bayesian model families.
- **Key Deliverables**:
  - **Frequentist HMM**: Gaussian and Student-$t$ emissions with automated post-fit label alignment to the 5 regimes.
  - **Bayesian HMM**: Dirichlet prior transitions, PyMC/NUTS sampling, Gelman-Rubin convergence checks.
  - **Regime-Switching VAR**: Multi-equation macro-equity dynamics across regimes.
  - **Bayesian Deep Learning**: Temporal ConvNet with Monte Carlo Dropout and Variational BNN (Bayes by Backprop via Pyro).
- **Validation Gate**: MCMC diagnostics ($\hat{R} < 1.05$, zero divergences); baseline out-of-sample log-loss evaluation.

### Phase 3: Foundation Adapter, Ensembling & Conformal Calibration (Days 8, 9, 11 Compressed)
- **Objective**: Integrate time-series foundation models, assemble the model federation, and apply distribution-robust conformal guarantees.
- **Key Deliverables**:
  - **Foundation Model Adapter**: Unified interface for Chronos and TimesFM/Moirai with CPU/GPU graceful degradation.
  - **Model Ensembling**: Bayesian Model Averaging (BMA) with PSIS-LOO weights + Constrained simplex stacking.
  - **Calibration Layer**: Adaptive Prediction Sets (APS) and Adaptive Conformal Inference (ACI) for non-exchangeable market series; reliability diagrams and ECE calculation.
- **Validation Gate**: Out-of-sample calibration audit; empirical coverage and set size documented; reliability diagrams generated across market cycles.

### Phase 4: Online Streaming, Risk Engine & Historical Backtest (Days 10, 12, 13 Compressed)
- **Objective**: Implement the intraday two-speed engine and execute the 2019–2024 allocation backtest.
- **Key Deliverables**:
  - **Two-Speed Online Engine**: Bootstrap Particle Filter (SMC) + Bayesian Online Changepoint Detection (BOCPD) with online/batch KL reconciliation.
  - **Regime-Conditioned Simulation**: Forward Monte Carlo paths, Value at Risk (VaR 95/99), Expected Shortfall (CVaR).
  - **2019–2024 Backtest**: Dynamic regime overlay vs Nifty 50 buy-and-hold, including transaction costs and turnover.
  - **Indian Case Studies**: In-depth empirical audits of COVID-19 crash (March 2020), 2021 post-shock rally, and 2024 election volatility.
- **Validation Gate**: Objective out-of-sample backtest metrics computed; performance compared against defined benchmarks (buy-and-hold Nifty 50, cash baseline); failure cases, stress periods, and drawdown behaviors documented without cherry-picking.

### Phase 5: R Reconciliation, Reporting & Handover (Days 14–15 Compressed)
- **Objective**: Complete the parallel R implementation, verify cross-language parity, and assemble the six mandatory deliverables.
- **Key Deliverables**:
  - **R Codebase**: `depmixS4` HMM, `MSwM` switching models, and numerical reconciliation script.
  - **Deliverable 1: Main Report** ($\ge 40$ pages, publication standard, theory to Investment Committee narrative).
  - **Deliverable 2 & 3**: Clean, fully typed Python and R codebases.
  - **Deliverable 4**: Backtesting and Simulation engine.
  - **Deliverable 5**: Model Card and Validation Pack.
  - **Deliverable 6**: 18-slide executive presentation and 10-minute demonstration video script.
- **Validation Gate**: Python-vs-R numerical difference $< 10^{-2}$; complete end-to-end reproducibility test passing.

---

## 4. Resource Allocation & Hardware Management

| Component | Compute Profile | Mitigation Strategy |
| :--- | :--- | :--- |
| **Bayesian HMM (MCMC)** | High CPU multi-core | Vectorized PyMC NUTS; parallel chains; caching posterior traces |
| **Foundation Models** | High GPU VRAM (if unquantized) | Quantized small checkpoints (e.g. `chronos-t5-small`); CPU fallback; unified adapter mock mode |
| **Variational BNN** | Moderate GPU / High CPU | Mini-batch ELBO optimization with AdamW |
| **Particle Filter (SMC)** | Moderate CPU vectorized | Vectorized NumPy particle resampling with systematic resample |
| **TDA Homology** | Moderate CPU | Subsample point clouds or use sliding window with fixed dimension $d \le 3$ |
