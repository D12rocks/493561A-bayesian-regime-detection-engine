# System Architecture Specification

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited  
> **Project:** Bayesian Regime Detection Engine for Equity Direction Forecasting

---

## 1. System Vision & Core Philosophy

Indian financial markets exhibit acute non-stationarity driven by macroeconomic shifts, global capital flows (FIIs), domestic structural liquidity (DIIs / mutual fund SIP inflows), currency fluctuations (USD/INR), and sovereign debt dynamics. Standard econometric linear models assume static parameters, leading to catastrophic failure during transition periods or structural breaks.

This architecture treats market state estimation not as a scalar price prediction problem, but as a **probabilistic hidden state inference problem**.

### Four Core Axioms
1. **Direction Over Price**: Predicting continuous asset price returns over short horizons is dominated by Brownian noise; predicting discrete macroscopic structural states (regimes) captures persistent trends and asymmetric volatility clusters.
2. **Probability Over Point Forecast**: The engine must yield full categorical probability distributions $\mathbf{p}_t = [p_1, \dots, p_5]^\top \in \Delta^4$ over states rather than hard classifications.
3. **Calibrated Uncertainty Over False Precision**: Point probabilities must be supplemented by conformal prediction sets $\mathcal{C}_{1-\alpha}(x_t) \subseteq \{1, \dots, 5\}$ guaranteeing marginal coverage even under non-exchangeable distribution shifts, accompanied by a decomposition of uncertainty into epistemic (model parameter ignorance) and aleatoric (inherent market entropy).
4. **Documented Ensemble Over Single Black Box**: No individual model has structural optimality across all market cycles. A calibrated ensemble integrating Bayesian Hidden Markov Models, Regime-Switching VARs, Variational Neural Networks, and Foundation Models provides robust generalization.

---

## 2. Mathematical Definition of the 5 Regimes

Let $S_t \in \{1, 2, 3, 4, 5\}$ be the discrete latent market state at date $t$. The regimes are formalized as follows:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           S_t State Space                              │
  ├───────────────┬──────────────┬──────────────┬─────────────┬────────────┤
  │  1. Risk-On   │ 2. Risk-Off  │ 3. Transition│ 4. Late-Cyc │ 5. Post-Shk│
  └───────────────┴──────────────┴──────────────┴─────────────┴────────────┘
```

### 1. Risk-On ($S_t = 1$)
- **Economic Context**: Sustained equity expansion, broad market participation, aggressive risk appetite, expanding liquidity.
- **Parametric Signatures**:
  - Expected Return: $\mathbb{E}[R_{t}^{\text{Nifty}}] > 0$
  - Realized Volatility: $\sigma_t^{\text{Nifty}} < \text{Quantile}_{0.35}(\sigma)$
  - India VIX: $\text{VIX}_t \le 15.0$
  - Market Breadth: $> 65\%$ of Nifty 50 constituents trading above their 50-day moving average.
  - Capital Flows: Cumulative Net FII + DII flows $> 0$ over rolling 20 days.

### 2. Risk-Off ($S_t = 2$)
- **Economic Context**: Deleveraging, liquidity contraction, flight to sovereign debt and US Dollar, severe drawdowns.
- **Parametric Signatures**:
  - Expected Return: $\mathbb{E}[R_{t}^{\text{Nifty}}] < 0$
  - Realized Volatility: Elevated, $\sigma_t^{\text{Nifty}} > \text{Quantile}_{0.70}(\sigma)$
  - India VIX: $\text{VIX}_t \ge 22.0$ with positive momentum ($\Delta \text{VIX} > 0$).
  - FX & Yields: Negative correlation with INR ($\Delta \text{USDINR} > 0$), widening corporate credit spreads.

### 3. Transitional ($S_t = 3$)
- **Economic Context**: Boundary state between macro cycles, characterized by conflicting signals, high entropy, and regime instability.
- **Parametric Signatures**:
  - State Entropy: $\mathcal{H}(\mathbf{p}_t) = -\sum_{k=1}^5 p_{tk} \log p_{tk} \to \max$.
  - Indicator Divergence: High index momentum alongside collapsing market breadth; or collapsing VIX amidst negative market returns.
  - Duration: Shorter expected state persistence $\frac{1}{1 - A_{33}}$ compared to Risk-On or Risk-Off.

### 4. Late-Cycle ($S_t = 4$)
- **Economic Context**: Mature expansion, narrow large-cap leadership, stretched valuation multiples, rising input costs, yield curve flattening.
- **Parametric Signatures**:
  - Relative Breadth: Large-cap outperforming Mid/Small-cap indices: $R_t^{\text{Nifty 50}} - R_t^{\text{Midcap 100}} > 0$ with declining overall constituent breadth.
  - Macro Signals: Flattening 10Y Indian Sovereign Gilt curve, decelerating monthly SIP growth rate.

### 5. Post-Shock ($S_t = 5$)
- **Economic Context**: Volatility compression following severe market drawdowns; institutional accumulation; mean-reversion onset.
- **Parametric Signatures**:
  - Volatility Dynamics: Absolute VIX remains high ($\text{VIX} > 20$), but first difference is strongly negative ($\frac{d}{dt}\text{VIX} \ll 0$).
  - Extreme Oversold Reversal: Relative Strength Index (14-day) rebounding from $< 30$ with positive divergence.

---

## 3. Comprehensive Model Stack

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    MODEL SUBSYSTEM                     │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
        ┌──────────────────────┬───────────────────────────┼───────────────────────────┬──────────────────────┐
        ▼                      ▼                           ▼                           ▼                      ▼
┌──────────────┐       ┌──────────────┐            ┌──────────────┐            ┌──────────────┐       ┌──────────────┐
│ Frequentist  │       │   Bayesian   │            │   Regime-    │            │   Bayesian   │       │  Foundation  │
│     HMM      │       │     HMM      │            │Switching VAR │            │Deep Learning │       │ Model Adapter│
├──────────────┤       ├──────────────┤            ├──────────────┤            ├──────────────┤       ├──────────────┤
│ Gaussian &   │       │ Dirichlet    │            │ Hamilton     │            │ • MC Dropout │       │ • Chronos    │
│ Student-t    │       │ Priors, MCMC │            │ Multi-var    │            │ • Variational│       │ • TimesFM /  │
│ emissions    │       │ NUTS / PyMC  │            │ macro states │            │   BNN (Pyro) │       │   Moirai     │
└──────────────┘       └──────────────┘            └──────────────┘            └──────────────┘       └──────────────┘
```

### 3.1 Frequentist Hidden Markov Model
- **Topology**: 5-state first-order Markov chain with Gaussian and Student-$t$ mixture emission distributions to account for fat tails in asset returns.
- **Parameter Estimation**: Baum-Welch (Expectation-Maximization) algorithm with regularization on covariance matrices to prevent degeneracy during crisis periods.
- **Post-Fit State Labeling**: Dynamic permutation optimization matching fitted states to the 5 semantic regimes based on sorted empirical characteristics:
  $$\pi^* = \arg\max_{\pi \in \mathcal{S}_5} \sum_{k=1}^5 \text{Score}(k, \pi(k))$$

### 3.2 Bayesian Hidden Markov Model
- **Probabilistic Formulation**:
  - Transition rows: $A_{j, \cdot} \sim \text{Dirichlet}(\alpha_{j, 1}, \dots, \alpha_{j, 5})$ with prior hyperparameter $\alpha_{j, j} > \alpha_{j, k}$ encoding state persistence.
  - Emission parameters:
    $$\mu_k \sim \mathcal{N}(\mu_0, \Sigma_0), \quad \Sigma_k \sim \text{Inverse-Wishart}(\nu_0, \Psi_0)$$
- **Sampling & Inference**: Hamiltonian Monte Carlo (HMC) / No-U-Turn Sampler (NUTS) implemented via PyMC and NumPyro.
- **Diagnostics**: Mandatory posterior convergence checks: Gelman-Rubin $\hat{R} < 1.05$, effective sample size $\text{ESS} > 400$, and zero energy divergences.

### 3.3 Regime-Switching Vector Autoregression (RS-VAR)
- **Formulation**: Vector of macro-equity signals $\mathbf{Y}_t \in \mathbb{R}^d$ governed by:
  $$\mathbf{Y}_t = \mathbf{c}_{S_t} + \sum_{p=1}^P \mathbf{\Phi}_{S_t, p} \mathbf{Y}_{t-p} + \mathbf{\epsilon}_t, \quad \mathbf{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{\Omega}_{S_t})$$
- Captures dynamic cross-asset transmission between Nifty returns, India VIX changes, Gilt yields, and USD/INR exchange rate across different states.

### 3.4 Bayesian Deep Learning
- **Architectures**:
  1. **Monte Carlo Dropout (MC Dropout)**: Deep temporal convolutional network with spatial dropout active at inference time. Posterior mean and epistemic variance evaluated via $T = 100$ stochastic forward passes:
     $$\hat{\mathbf{p}}_t = \frac{1}{T}\sum_{\tau=1}^T \text{Softmax}(f_{\hat{\mathbf{W}}_\tau}(\mathbf{x}_t))$$
  2. **Variational BNN (Bayes by Backprop)**: Stochastic neural network placing Gaussian priors $\mathcal{N}(0, \sigma_0^2)$ on weights, optimized via Evidence Lower Bound (ELBO) maximization using Pyro/PyTorch.
  3. **Deep Ensemble**: Multi-seed initialized networks with randomized training partitions for non-Bayesian epistemic diversity.

### 3.5 Time-Series Foundation Models
- **Integration Principle**: Foundation models pre-trained on billions of time-series points (e.g., Chronos based on T5, TimesFM, Lag-Llama, Moirai) provide zero-shot probabilistic trajectory representations.
- **Unified Adapter Interface (`FoundationModelAdapter`)**:
  - Extracts latent embeddings or zero-shot quantile forecasts without modifying downstream architecture.
  - A lightweight Bayesian multinomial logistic regression head maps foundation embeddings to the 5 regime probabilities.
  - Explicit hardware fallback: if local GPU resources are insufficient for large foundation checkpoints, a documented CPU quantized runtime or documented mock-interface is employed without fabricating embeddings.

---

## 4. Uncertainty Quantification & Decomposition

For any observation $\mathbf{x}_t$ and model output $\mathbf{p}_t = [p_{t1}, \dots, p_{t5}]$, total predictive uncertainty is formally decomposed:

```
                               ┌────────────────────────────────────────────────────────┐
                               │              Total Predictive Uncertainty              │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                                                     ▼
                  ┌───────────────────────────┐                         ┌───────────────────────────┐
                  │   Aleatoric Uncertainty   │                         │   Epistemic Uncertainty   │
                  │  (Irreducible Data Noise) │                         │   (Model/Parameter Lack)  │
                  ├───────────────────────────┤                         ├───────────────────────────┤
                  │ Average Predictive        │                         │ Mutual Information /      │
                  │ Shannon Entropy           │                         │ Posterior Parameter Var   │
                  │ H_aleatoric = E[H(p_t)]   │                         │ I_epistemic = H(E[p])-H_al│
                  └───────────────────────────┘                         └───────────────────────────┘
```

1. **Aleatoric Uncertainty** (Irreducible stochasticity in market regime transitions):
   $$\mathcal{U}_{\text{aleatoric}}(\mathbf{x}_t) = \frac{1}{M}\sum_{m=1}^M \mathcal{H}(\mathbf{p}_t^{(m)}) = -\frac{1}{M}\sum_{m=1}^M \sum_{k=1}^5 p_{tk}^{(m)} \log p_{tk}^{(m)}$$
2. **Epistemic Uncertainty** (Model ignorance due to limited training data in specific regimes):
   $$\mathcal{U}_{\text{epistemic}}(\mathbf{x}_t) = \mathcal{H}\left(\frac{1}{M}\sum_{m=1}^M \mathbf{p}_t^{(m)}\right) - \mathcal{U}_{\text{aleatoric}}(\mathbf{x}_t)$$

---

## 5. Model Ensembling Engine

The ensemble combines $M$ calibrated models using two complementary methodologies:

```
  Models: [HMM, Bayesian HMM, RS-VAR, BNN, Foundation Head]
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │             Hurdle & Audit Check:              │
        │ • Validated out-of-sample calibration          │
        │ • Gelman-Rubin R-hat < 1.05 (for MCMC)         │
        │ • Maximum permitted ECE < 0.12                 │
        └────────────────────────┬───────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │   Bayesian Model    │     │ Constrained Stacking│
        │   Averaging (BMA)   │     │ (Dirichlet Prior)   │
        ├─────────────────────┤     ├─────────────────────┤
        │ Weights proportional│     │ Super-learner QP    │
        │ to marginal likeli- │     │ on simplex:         │
        │ hood or PSIS-LOO    │     │ w_i >= 0, sum w = 1 │
        └──────────┬──────────┘     └──────────┬──────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                     Final Ensemble Probability
```

1. **Bayesian Model Averaging (BMA)**:
   $$w_m = \frac{p(\mathcal{D} \mid \mathcal{M}_m) p(\mathcal{M}_m)}{\sum_{j=1}^M p(\mathcal{D} \mid \mathcal{M}_j) p(\mathcal{M}_j)}$$
   Where marginal likelihood $p(\mathcal{D} \mid \mathcal{M}_m)$ is computed via thermodynamic integration or approximated via Pareto-Smoothed Importance Sampling LOO (PSIS-LOO / WAIC).
2. **Constrained Stacking**:
   Solves a convex optimization on the probability simplex to minimize log-loss or Brier score on the calibration set:
   $$\mathbf{w}^* = \arg\min_{\mathbf{w} \in \Delta^{M-1}} -\sum_{t \in \mathcal{T}_{\text{cal}}} \sum_{k=1}^5 y_{tk} \log\left(\sum_{m=1}^M w_m p_{tk}^{(m)}\right)$$

---

## 6. Conformal Prediction & Calibration Layer

Standard split conformal methods assume exchangeability ($\text{i.i.d.}$ observations), an assumption severely violated by financial time series displaying autocorrelation and volatility clustering.

### Conformal Prediction Architecture
1. **Adaptive Prediction Sets (APS)**:
   Sort regime probabilities: $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(5)}$. Include regimes until cumulative probability exceeds calibrated threshold:
   $$\mathcal{C}_{1-\alpha}(\mathbf{x}_t) = \left\{k : \sum_{j=1}^{\text{rank}(k)} p_{(j)}(\mathbf{x}_t) \le \hat{q}_{1-\alpha} \right\}$$
2. **Distribution-Shift-Robust Conformal**:
   - **Adaptive Conformal Inference (ACI)**: Dynamically updates the nominal error rate $\alpha_t$ based on recent empirical coverage errors:
     $$\alpha_{t+1} = \alpha_t + \gamma (\alpha - \text{err}_t), \quad \text{err}_t = \mathbb{I}(y_t \notin \mathcal{C}_{\alpha_t}(\mathbf{x}_t))$$
   - Guarantees long-run valid empirical coverage even under regime shifts and volatility bursts.
3. **Calibration Metrics**:
   - Reliability diagrams with $B = 10$ bins.
   - Expected Calibration Error (ECE):
     $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
   - Multi-class Brier Score and logarithmic loss.

---

## 7. Two-Speed Online Inference Engine

To support institutional deployment, the engine separates slow, compute-intensive Bayesian batch retraining from millisecond intraday streaming inference:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TWO-SPEED ARCHITECTURE                           │
├──────────────────────────────────────┬──────────────────────────────────────┤
│         NIGHTLY BATCH LAYER          │        INTRADAY STREAMING LAYER      │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Full MCMC sampling (PyMC/NumPyro)  │ • Bootstrap Particle Filter (SMC)    │
│ • RS-VAR parameter re-estimation     │ • Bayesian Online Changepoint        │
│ • Hyperparameter recalibration       │   Detection (BOCPD - Adams & MacKay) │
│ • Prior updates from empirical Bayes │ • Streaming Dirichlet-Multinomial    │
│ • Execution budget: 30-60 minutes    │ • Execution budget: < 10 milliseconds│
└──────────────────────────────────────┴──────────────────────────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │    Online / Batch Reconciliation     │
               ├──────────────────────────────────────┤
               │ Kullback-Leibler divergence audit:   │
               │   D_KL(p_online || p_batch_ref) < eps│
               │ Triggers alert on structural rupture │
               └──────────────────────────────────────┘
```

---

## 8. Topological Data Analysis (TDA) & Graph Neural Network (GNN) Features

- **TDA Feature Extraction**:
  - Time-delay embedding of Nifty returns and multi-asset price vectors into point clouds.
  - Vietoris-Rips filtration to compute persistent homology ($H_0, H_1$).
  - Extracted features: persistence entropy, total persistence, and persistence landscape norms, capturing topological market turbulence prior to crashes.
- **Sector Graph Representation (GNN)**:
  - Dynamic graph $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$ where nodes $\mathcal{V}$ represent 11 NSE sector indices (Nifty Bank, Nifty IT, Nifty Pharma, etc.).
  - Edge weights $e_{ij, t}$ represent rolling partial correlations or Granger causality.
  - Lightweight Graph Convolutional Network (GCN) embeds the cross-sector correlation topology into a low-dimensional state vector.

---

## 9. Backtesting, Simulation & Risk Engine

```
                             Regime Prediction Output
                                        │
                                        ▼
             ┌────────────────────────────────────────────────────┐
             │       Dynamic Allocation Overlay Engine            │
             │ (Risk-On: 100% Equity, Risk-Off: 0% + Hedging,     │
             │  Transitional: 40%, Late-Cycle: 60% Defensive,     │
             │  Post-Shock: Tactical Re-entry)                    │
             └─────────────────────────┬──────────────────────────┘
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
             ┌──────────────────────┐      ┌──────────────────────┐
             │ 2019-2024 Historical │      │  Regime-Conditioned  │
             │ Walk-Forward Backtest│      │  Monte Carlo (Paths) │
             ├──────────────────────┤      ├──────────────────────┤
             │ • Information Ratio  │      │ • Transition-guided  │
             │ • Max Drawdown vs B&H│      │   path simulation    │
             │ • Transaction Costs  │      │ • Forward VaR / CVaR │
             │ • Slippage Model     │      │   (95% & 99% levels) │
             └──────────┬───────────┘      └──────────┬───────────┘
                        │                             │
                        └──────────────┬──────────────┘
                                       ▼
                         Investment Committee Package
                         (Executive Brief, Risk Sheet)
```

---

## 10. Dual-Language Architecture & R Reconciliation

The engine maintains a formal parallel R research layer:
- **R Model Suite**: `depmixS4` for HMMs, `MSwM` for Markov-switching regression, and `changepoint` for BOCPD cross-validation.
- **Reconciliation Protocol**: Automated scripts run identical test vectors through both Python and R implementations, verifying:
  1. Mean absolute difference in regime probabilities: $\frac{1}{N}\sum |p_{\text{Python}} - p_{\text{R}}| < 10^{-3}$.
  2. Transition probability matrix Frobenius norm divergence: $\|A_{\text{Python}} - A_{\text{R}}\|_F < 10^{-2}$.
  3. VaR / CVaR numeric parity within numerical tolerance.
