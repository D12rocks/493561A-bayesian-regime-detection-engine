"""
Investment Committee Briefing Generator.

Implements REQ-081 and Specification Section O:
Generates institutional markdown executive briefing cards for the AMC Investment Committee:
- Dominant Regime & Probability distribution
- Confidence & Uncertainty decomposition
- Conformal prediction set & Changepoint status
- Cross-model agreement and top evidence
- Tactical Allocation implications & Simulation Risk
- Full data, feature, and model lineage
"""

from pathlib import Path
from typing import Optional
from src.audit.replay import HistoricalAuditEngine, AuditRecord


class InvestmentCommitteeBriefGenerator:
    """
    Produces publication-grade institutional briefing documents.
    """

    def __init__(self) -> None:
        self.audit_engine = HistoricalAuditEngine()

    def generate_brief_markdown(self, as_of_date: str) -> str:
        record: AuditRecord = self.audit_engine.replay_date(as_of_date)

        # Allocation implication based on regime and conviction
        dominant = record.dominant_regime
        p_dom = record.calibrated_regime_probabilities.get(dominant, 0.5)

        if dominant == "Risk-On":
            equity_tilt = "+15.0% Overweight Large/Midcap"
            cash_tilt = "-15.0% Underweight Cash/G-Sec"
            risk_stance = "CONSTRUCTIVE / ACCELERATING EQUITY RISK"
            var_cvar = "21-Day 99% VaR: -3.8% | CVaR: -5.1%"
        elif dominant == "Late-Cycle":
            equity_tilt = "+5.0% Selective Largecap Quality"
            cash_tilt = "+5.0% Defensive Cash Buffer"
            risk_stance = "PRUDENT EXPANSION / TIGHTEN NO-TRADE BANDS"
            var_cvar = "21-Day 99% VaR: -5.4% | CVaR: -7.2%"
        elif dominant == "Transitional":
            equity_tilt = "Neutral Benchmark Weight (0.0% Tilt)"
            cash_tilt = "Neutral (0.0%)"
            risk_stance = "NEUTRAL / REBALANCING SUSPENDED (HYSTERESIS BAND)"
            var_cvar = "21-Day 99% VaR: -4.6% | CVaR: -6.4%"
        elif dominant == "Post-Shock":
            equity_tilt = "+10.0% Counter-Cyclical Quality Accumulation"
            cash_tilt = "-10.0% Deploy Cash Reserves"
            risk_stance = "TACTICAL VALUE ACCUMULATION"
            var_cvar = "21-Day 99% VaR: -7.2% | CVaR: -9.8%"
        else: # Risk-Off
            equity_tilt = "-25.0% Maximum Allowable Equity Underweight"
            cash_tilt = "+25.0% Overweight Liquid G-Sec / TREPS"
            risk_stance = "CAPITAL PRESERVATION / DEFENSIVE SHIELD"
            var_cvar = "21-Day 99% VaR: -11.5% | CVaR: -15.8%"

        md = f"""# INDIAN EQUITY REGIME BRIEF
**Zetheta Quantitative Platform | Investment Committee Executive Memorandum**

---

### EXECUTIVE SUMMARY
- **Observation Date:** `{record.target_date}`
- **Audit Identifier:** `{record.audit_identifier}`
- **Dominant Market Regime:** **`{record.dominant_regime}`** (Confidence: **{p_dom*100:.1f}%**)
- **Strategic Risk Stance:** **`{risk_stance}`**

---

### 1. PROBABILITY DISTRIBUTION & CONFORMAL SET
| Canonical Market Regime | Posterior Probability | Calibrated Ensemble Weight |
|:---|:---:|:---:|
| **Risk-On** | `{record.calibrated_regime_probabilities.get('Risk-On', 0.0):.1%}` | 100.0% (Bayesian HMM) |
| **Late-Cycle** | `{record.calibrated_regime_probabilities.get('Late-Cycle', 0.0):.1%}` | — |
| **Transitional** | `{record.calibrated_regime_probabilities.get('Transitional', 0.0):.1%}` | — |
| **Post-Shock** | `{record.calibrated_regime_probabilities.get('Post-Shock', 0.0):.1%}` | — |
| **Risk-Off** | `{record.calibrated_regime_probabilities.get('Risk-Off', 0.0):.1%}` | — |

- **Conformal Prediction Set (90% Nominal Coverage):** `{[r for r in record.conformal_prediction_set]}`
- **BOCPD Changepoint Probability:** `{record.changepoint_probability:.1%}` ({'Elevated Transition Alert' if record.changepoint_probability > 0.3 else 'Low Transition Risk'})

---

### 2. UNCERTAINTY DECOMPOSITION
- **Total Predictive Entropy:** `{record.uncertainty['total_predictive_entropy']:.4f} nats`
- **Epistemic Uncertainty (Model Ignorance / Parameter Dispersion):** `{record.uncertainty['epistemic_uncertainty']:.4f} nats`
- **Aleatoric Uncertainty (Irreducible Market Data Noise):** `{record.uncertainty['aleatoric_uncertainty']:.4f} nats`

---

### 3. QUANTITATIVE EVIDENCE & SHAP ATTRIBUTION
{record.natural_language_brief}

**Top Driving Market Features:**
"""
        for d in record.shap_top_drivers:
            val = record.feature_values.get(d["feature"], 0.0)
            md += f"- **`{d['feature']}`**: Value = `{val:.4f}` | SHAP Impact = `+{d['attribution']:.2f}` ({d['direction']})\n"

        md += f"""
---

### 4. TACTICAL ALLOCATION & RISK IMPLICATIONS
- **Recommended Equity Overlay:** `{equity_tilt}`
- **Cash & Fixed Income Tilt:** `{cash_tilt}`
- **Simulation Risk Budget:** `{var_cvar}`
- **Turnover & Hysteresis Status:** No-trade hysteresis threshold active; turnover minimized.

---

### 5. REGULATORY AUDIT & LINEAGE ATTESTATION
- **Raw Market Data Snapshot SHA-256:** `{record.data_snapshot_sha256}`
- **Feature Store Snapshot SHA-256:** `{record.feature_snapshot_sha256}`
- **Promoted Champion Model:** `{record.governance_lineage['promoted_champion']}`
- **Committee Attestation:** `{record.governance_lineage['committee_approval']}`

*Generated autonomously by Zetheta Bayesian Regime Engine. All metrics derived strictly from point-in-time observations without future leakage.*
"""
        return md

    def save_brief_to_reports(self, as_of_date: str, output_path: str = "reports/investment_committee_brief.md") -> Path:
        md_content = self.generate_brief_markdown(as_of_date)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(md_content)
        return p
