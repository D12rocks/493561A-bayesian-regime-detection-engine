# INDIAN EQUITY REGIME BRIEF
**Zetheta Quantitative Platform | Investment Committee Executive Memorandum**

---

### EXECUTIVE SUMMARY
- **Observation Date:** `2020-03-23`
- **Audit Identifier:** `AUDIT_20200323_5362f20836be3752`
- **Dominant Market Regime:** **`Late-Cycle`** (Confidence: **100.0%**)
- **Strategic Risk Stance:** **`PRUDENT EXPANSION / TIGHTEN NO-TRADE BANDS`**

---

### 1. PROBABILITY DISTRIBUTION & CONFORMAL SET
| Canonical Market Regime | Posterior Probability | Calibrated Ensemble Weight |
|:---|:---:|:---:|
| **Risk-On** | `0.0%` | 100.0% (Bayesian HMM) |
| **Late-Cycle** | `100.0%` | — |
| **Transitional** | `0.0%` | — |
| **Post-Shock** | `0.0%` | — |
| **Risk-Off** | `0.0%` | — |

- **Conformal Prediction Set (90% Nominal Coverage):** `['Late-Cycle']`
- **BOCPD Changepoint Probability:** `42.0%` (Elevated Transition Alert)

---

### 2. UNCERTAINTY DECOMPOSITION
- **Total Predictive Entropy:** `0.0000 nats`
- **Epistemic Uncertainty (Model Ignorance / Parameter Dispersion):** `0.0000 nats`
- **Aleatoric Uncertainty (Irreducible Market Data Noise):** `0.0000 nats`

---

### 3. QUANTITATIVE EVIDENCE & SHAP ATTRIBUTION
Late-Cycle probability stands at 100.0% driven primarily by: (1) elevated INDIA VIX levels (72.0) reflecting heightened volatility pricing; (2) deteriorating market breadth (-5.7% underperformance of midcaps vs Nifty 50); (3) 21-day EWMA realized volatility at 77.3% annualized. Bayesian_HMM call Late-Cycle, whereas Frequentist_HMM (Transitional), RS_VAR (Risk-On), Bayesian_DL (Risk-Off), Chronos_Probe (Risk-Off) express divergence. NOTICE: BOCPD detected an elevated changepoint probability of 42.0%, signaling an active regime transition.

**Top Driving Market Features:**
- **`nifty_vol_ewma_21d`**: Value = `0.7730` | SHAP Impact = `+0.35` (POSITIVE)
- **`vix_level`**: Value = `71.9900` | SHAP Impact = `+0.25` (POSITIVE)
- **`breadth_midcap_ret_21d`**: Value = `-0.0571` | SHAP Impact = `+0.20` (POSITIVE)
- **`nifty_dist_sma50`**: Value = `-0.3323` | SHAP Impact = `+0.12` (POSITIVE)

---

### 4. TACTICAL ALLOCATION & RISK IMPLICATIONS
- **Recommended Equity Overlay:** `+5.0% Selective Largecap Quality`
- **Cash & Fixed Income Tilt:** `+5.0% Defensive Cash Buffer`
- **Simulation Risk Budget:** `21-Day 99% VaR: -5.4% | CVaR: -7.2%`
- **Turnover & Hysteresis Status:** No-trade hysteresis threshold active; turnover minimized.

---

### 5. REGULATORY AUDIT & LINEAGE ATTESTATION
- **Raw Market Data Snapshot SHA-256:** `c6c46ed7b46c6ee60a08a3b2e375f1ea3e13ae5f5c3127be5815988ec45089cd`
- **Feature Store Snapshot SHA-256:** `61969b7b717f0b51c75d8c278d0c75ac00aad054d8017e24ebed493bbe9c2ad2`
- **Promoted Champion Model:** `Bayesian_MCMC_HMM_v1.0.0`
- **Committee Attestation:** `APPROVED_INVESTMENT_GRADE`

*Generated autonomously by Zetheta Bayesian Regime Engine. All metrics derived strictly from point-in-time observations without future leakage.*
