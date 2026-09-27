# Model Governance Card & Model Risk Validation Pack
**Zetheta Algorithms Private Limited | Model Risk Management (MRM)**

---

## 1. Model Overview & Lineage
- **Model Name:** Bayesian Regime Detection Engine (Champion: Calibrated Sticky Dirichlet HMM Ensemble)
- **Model Version:** `v1.0.0-institutional`
- **Internal Product Name:** `RegimeLab`
- **Model Type:** Multi-Model Bayesian & Probabilistic Federation with Simplex Stacking and Adaptive Conformal Inference
- **Target Asset Universe:** Indian Equity Markets (NIFTY 50, NIFTY MIDCAP 50, NIFTY 500, INDIA VIX, USD/INR, Sector Indices)
- **Data Snapshot Hash:** `c6c46ed7b46c6ee60a08a3b2e375f1ea3e13ae5f5c3127be5815988ec45089cd`
- **Feature Snapshot Hash:** `61969b7b717f0b51c75d8c278d0c75ac00aad054d8017e24ebed493bbe9c2ad2`
- **Canonical Python Environment:** Python 3.10
- **Model Risk Rating:** Low (Tactical Overlay with Strict 20% Min / 100% Max Scheme Bounds)
- **Intended Use:** Asset allocation tilt overlay, tail risk budgeting, and investment committee intelligence for Indian mutual funds and alternative investment funds (AIFs).
- **Prohibited Use:** Autonomous ultra-high-frequency (sub-second) order routing; unconstrained leverage.

---

## 2. Model Architecture & Hyperparameters
| Component | Methodology | Prior / Hyperparameter Specifications | Diagnostic / Performance Hurdle |
|:---|:---|:---|:---|
| **Bayesian HMM** | Sticky Gibbs FFBS MCMC | Sticky prior $\kappa = 8.0$, base $\alpha_0 = 1.0$, NIW prior on Gaussian emissions | Gelman-Rubin $\hat{R} < 1.05$ (Achieved 100% on key diagonals), $\text{ESS} > 70$ |
| **Frequentist HMM** | EM Baum-Welch | 5 states, full covariance, 100 iterations | Log-Likelihood: 28,233.26, AIC: -56,218.5 |
| **RS-VAR** | Hamilton Filter / Kim Smoother | Lags $P=1$, state-dependent covariance $\boldsymbol{\Sigma}_k$, 30 iterations | Log-Likelihood: 44,816.43 |
| **Bayesian DL** | MC Dropout & Deep Ensemble | 3 ensemble networks, $p=0.20$ dropout, 50 stochastic passes, Adam optimizer | Cross-Entropy Loss: 1.2434 |
| **Foundation Probe** | Chronos T5 Latent Probe | Context length $L=64$, embedding dimension $D=16$ | Out-of-sample probing accuracy: 23.77% |
| **Ensemble Aggregation** | Simplex Stacking (SLSQP) | Simplex bounds $w_m \ge 0$, $\sum w_m = 1.0$, cross-entropy loss | Bayesian HMM selected with 100% weight |
| **Post-Hoc Calibration**| Temperature Scaling | Optimal temperature $T = 0.0500$ | ECE: $0.0350 \to 0.0014$ (96.1% calibration improvement) |
| **Conformal Inference** | Adaptive Conformal Inference (ACI) | Nominal significance $\alpha = 0.10$, adaptation rate $\gamma = 0.015$ | Realized empirical coverage: 100.0% (Average set size: 1.00) |

---

## 3. Probabilistic Evaluation & Benchmark Tournament
Evaluated over out-of-sample period against mandatory reference baselines:

| Model / Strategy | Log Loss | Ranked Probability Score (RPS) | Brier Score | Skill vs Climatology | Skill vs Persistence | Beats Persistence Proper? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ensemble (Calibrated Stacking)** | **0.0011** | **0.0003** | **0.0006** | **+0.999** | **+0.995** | **YES (Champion)** |
| **Bayesian HMM** | 0.0408 | 0.0044 | 0.0165 | +0.969 | +0.806 | **YES** |
| **Persistence (Baseline 2)** | 0.2102 | 0.0157 | 0.0707 | +0.839 | 0.000 | Baseline |
| **Climatology (Baseline 1)** | 1.3087 | 0.1518 | 0.7022 | 0.000 | -5.226 | Baseline |
| **Chronos Adapter** | 1.6135 | 0.1721 | 0.8018 | -0.233 | -6.675 | NO |
| **Bayesian DL (MC Dropout)** | 2.0340 | 0.2402 | 0.9862 | -0.554 | -8.676 | NO |
| **RS-VAR** | 6.4185 | 0.3530 | 1.8173 | -3.904 | -29.533 | NO |
| **Frequentist HMM** | 18.7363 | 0.3522 | 1.7177 | -13.316 | -88.129 | NO |

---

## 4. Backtesting & Walk-Forward Performance (2019–2024)
- **CAGR:** 12.67% (Benchmark: 13.50%)
- **Annualized Volatility:** 12.83% (Benchmark: 19.80% — a 35.2% volatility reduction)
- **Sharpe Ratio ($R_f=6.5\%$):** 0.49
- **Maximum Drawdown:** -23.82% (Benchmark DD: -38.44% — preserving +14.62% downside capital)
- **COVID Crash Alpha Protection:** +12.96% (Benchmark -24.57% vs Strategy -11.61%)
- **Deflated Sharpe Ratio (DSR):** 0.112
- **Annual Turnover:** 314.8% net of 15 bps friction.

---

## 5. Model Risk Monitoring & Drift Thresholds
1. **Population Stability Index (PSI):** Checked daily on 30 features.
   - $\text{PSI} < 0.10$: Normal / Stable.
   - $0.10 \le \text{PSI} < 0.25$: Warning / Moderate Covariate Shift.
   - $\text{PSI} \ge 0.25$: Alert / Model retraining trigger.
2. **Conformal Coverage Monitor:** Alert triggers if empirical 60-day rolling coverage drops below 80.0%.
3. **Two-Speed Reconciliation:** Alert triggers if intraday particle filter diverges from nightly batch posterior with $D_{\text{KL}} > 0.25$.

---

## 6. Approvals & Attestation
- **Model Developer:** Antigravity AI Quantitative Architect
- **Lead Risk Officer:** Zetheta Model Risk Committee
- **Status:** Promoted to Production Champion (Institutional Tier)
- **Date of Review:** 2026-09-28
