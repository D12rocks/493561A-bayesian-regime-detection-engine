# Validation Protocol & Leakage Prevention Standard

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited  
> **Project:** Bayesian Regime Detection Engine for Equity Direction Forecasting

---

## 1. Principles of Financial Time-Series Validation

Ordinary machine learning practices—such as shuffled K-Fold cross-validation, whole-dataset standard scaling, or evaluating overlapping targets—are fatal to financial research. They introduce severe look-ahead bias and autocorrelation leakage, creating overly optimistic results that fail in production.

This protocol mandates **strict temporal isolation**, **adversarial leakage checks**, **formal MCMC diagnostic hurdles**, and **cross-language numerical reconciliation**.

---

## 2. Chronological Split Specification

All historical research in this project respects the following chronological partitioning:

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                   TIME HORIZON (2009 - 2024)                          │
├────────────────────────────────────────┬─────────────────────┬────────────────────────┤
│          TRAINING PERIOD               │  CALIBRATION PERIOD │       TEST PERIOD      │
│      2009-01-01 to 2017-12-31          │ 2018-01-01 to 2018- │ 2019-01-01 to 2024-    │
│            (~9 Years)                  │   12-31 (~1 Year)   │    12-31 (~6 Years)    │
├────────────────────────────────────────┼─────────────────────┼────────────────────────┤
│ • Model parameter estimation           │ • Conformal non-    │ • Out-of-sample back-  │
│ • MCMC posterior sampling              │   conformity scores │   testing & allocation │
│ • Prior elicitation                    │ • Stacking weights  │ • ECE & Brier audit    │
│ • Feature selection & transformations  │ • ECE calibration   │ • Indian Crisis Case   │
│                                        │ • Threshold tuning  │   Studies (COVID, etc.)│
└────────────────────────────────────────┴─────────────────────┴────────────────────────┘
```

### Expanding Walk-Forward Protocol
For production simulation and rolling validation, an **expanding window** or **anchored walk-forward** structure is enforced:
- **Initial Training Window**: 5 years minimum.
- **Calibration Window**: Rolling 252 trading days preceding the test day.
- **Step Size**: Monthly or quarterly recalibration of batch models.
- **Never Tune on Test**: Any model parameter, prior, or stacking weight tuned on the test period invalidates the experiment.

---

## 3. Automated Leakage Prevention Framework

The test suite incorporates automated checks to verify the absence of all six major forms of leakage:

| Leakage Category | Manifestation in Code | Enforcement Mechanism |
| :--- | :--- | :--- |
| **1. Look-Ahead Bias** | Computing indicators using $t+1, \dots, t+k$ observations | Assert that for any feature computed at timestamp $T$, altering observations at $T+1$ causes zero change in the feature matrix up to $T$. |
| **2. Normalization Leakage** | Calling `StandardScaler.fit()` or computing min/max across the entire series | Enforce `fit()` on training subset only, or use historical rolling z-scores with explicit lag: $(x_t - \mu_{t-1:\text{hist}}) / \sigma_{t-1:\text{hist}}$. |
| **3. Macro Release Lag Leakage** | Treating monthly SIP or GDP numbers as available on the last day of the reference month | Shift macro series forward by the verified historical publication delay (e.g., SIPs shifted by +10 days). |
| **4. Overlapping Target Leakage** | Training a multi-day directional model on daily steps without purging | Apply Lopez de Prado purging and embargoing between train and calibration folds. |
| **5. Survivorship Bias** | Filtering index constituents based on their current inclusion | Ingest broad market universes with explicit delisting treatment and index reconstitution records. |
| **6. Stale Observation Bias** | Forward-filling illiquid securities indefinitely | Restrict maximum allowable forward-fill to 3 trading days; beyond that, drop or mark as untradable. |

---

## 4. Conformal Prediction & The Exchangeability Limitation

### The Exchangeability Dilemma
Conformal prediction theoretically guarantees marginal coverage $1 - \alpha$ under the assumption that calibration and test data points are **exchangeable** (e.g., independent and identically distributed). In financial markets:
$$\{ (x_t, y_t) \}_{t=1}^T \text{ is non-exchangeable due to serial correlation, volatility clustering, and macro regime shifts.}$$

### Mandatory Mitigations
1. **Adaptive Conformal Inference (ACI)**:
   The nominal significance level $\alpha_t$ must adapt dynamically to local coverage errors:
   $$\alpha_{t+1} = \alpha_t + \gamma (\alpha - \text{err}_t)$$
   where $\gamma \in [0.005, 0.05]$ is the learning rate and $\text{err}_t = \mathbb{I}(y_t \notin \mathcal{C}_{\alpha_t}(x_t))$.
2. **Mondrian / Group-Balanced Conformal**:
   Non-conformity scores are calibrated conditionally within high-volatility vs low-volatility partitions to ensure coverage does not collapse precisely when volatility spikes.

---

## 5. Bayesian MCMC Posterior Diagnostic Standards

A Bayesian model (such as the Bayesian HMM or BNN) is **never** considered valid merely because the sampling algorithm finished without raising an exception. Every MCMC run must satisfy:

1. **Gelman-Rubin Diagnostic ($\hat{R}$)**:
   $$\hat{R} < 1.05 \quad \text{for all latent and structural parameters.}$$
2. **Effective Sample Size (ESS)**:
   $$\text{Bulk-ESS} > 400 \quad \text{and} \quad \text{Tail-ESS} > 400.$$
3. **Divergent Transitions**:
   Zero numerical divergences during NUTS sampling. Any divergence indicates regions of high posterior curvature that the sampler failed to explore.
4. **Energy Bayesian Fraction of Missing Information (E-BFMI)**:
   $$\text{E-BFMI} > 0.3 \quad \text{for all sampling chains.}$$

---

## 6. Post-Fit Latent State Identification Protocol

Hidden Markov Models are subject to label switching: state $0$ in run A may correspond to state $2$ in run B. To bind latent states to the five canonical regimes:

1. Extract posterior mean return vector $\boldsymbol{\mu} \in \mathbb{R}^5$ and covariance/volatility vector $\boldsymbol{\sigma} \in \mathbb{R}^5$ across the 5 states.
2. Compute the state identification matrix:
   - State with lowest return and highest volatility $\to$ **Risk-Off**.
   - State with highest return and lowest volatility $\to$ **Risk-On**.
   - State with high volatility but negative volatility acceleration $\to$ **Post-Shock**.
   - State with moderate positive return but decelerating breadth / narrow leadership $\to$ **Late-Cycle**.
   - Remaining intermediate / high-entropy state $\to$ **Transitional**.
3. Persist the state-to-regime permutation map in the experiment metadata.

---

## 7. Dual-Language Python vs R Reconciliation Protocol

To verify mathematical consistency between Python and R implementations:

1. **Deterministic Benchmark Dataset**: Export an immutable 500-day evaluation vector of returns, volatility, and breadth (`tests/fixtures/reconciliation_vector.csv`).
2. **Execution**:
   - Run Python Frequentist HMM (`hmmlearn` / custom EM).
   - Run R HMM (`depmixS4`).
3. **Acceptance Thresholds**:
   - Log-likelihood convergence parity: $|\ln \mathcal{L}_{\text{Python}} - \ln \mathcal{L}_{\text{R}}| / |\ln \mathcal{L}_{\text{Python}}| < 0.01$.
   - Transition probability matrix Frobenius norm difference: $\|A_{\text{Python}} - A_{\text{R}}\|_F < 0.05$.
   - Regime probability correlation: Pearson $r > 0.95$ across all 5 state probabilities.
