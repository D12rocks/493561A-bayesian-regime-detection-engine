"""
Investment Committee Pack & Institutional Reporting Generator.

Generates executive briefings, risk ribbons, tail-risk breakdowns,
and dynamic allocation directives for portfolio managers and risk committees.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from src.models.contracts import RegimePrediction
from src.simulation.base import TailRiskMetrics


@dataclass
class InvestmentCommitteeBriefing:
    """Standardized institutional reporting dossier."""
    as_of_date: datetime
    current_regime: str
    regime_confidence: float
    conformal_prediction_set: List[str]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    changepoint_probability: float
    recommended_equity_exposure: float
    recommended_cash_hedge: float
    forward_var_95: float
    forward_cvar_95: float
    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date.strftime("%Y-%m-%d"),
            "current_regime": self.current_regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "conformal_prediction_set": self.conformal_prediction_set,
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "aleatoric_uncertainty": round(self.aleatoric_uncertainty, 4),
            "changepoint_probability": round(self.changepoint_probability, 4),
            "recommended_equity_exposure": round(self.recommended_equity_exposure, 2),
            "recommended_cash_hedge": round(self.recommended_cash_hedge, 2),
            "forward_var_95": round(self.forward_var_95, 4),
            "forward_cvar_95": round(self.forward_cvar_95, 4),
            "commentary": self.commentary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class InvestmentCommitteePackGenerator:
    """Generates complete reporting artifacts for executive decision-makers."""

    def generate_briefing(
        self,
        prediction: RegimePrediction,
        tail_risk: TailRiskMetrics,
        allocation_weights: Dict[str, float],
    ) -> InvestmentCommitteeBriefing:
        # Determine narrative commentary based on regime and uncertainty
        probs = prediction.regime_probabilities
        pred_regime = prediction.predicted_regime.value
        top_prob = getattr(probs, pred_regime.lower().replace("-", "_"))

        if prediction.uncertainty.epistemic_uncertainty > 0.4:
            commentary = (
                f"Elevated Epistemic Uncertainty ({prediction.uncertainty.epistemic_uncertainty:.2f}): "
                f"Models exhibit substantial dispersion. Conformal prediction set encompasses "
                f"{prediction.conformal_prediction_set}. Conservative risk posture recommended."
            )
        elif pred_regime == "Risk-Off":
            commentary = (
                f"Defensive Posture Activated: High conviction Risk-Off regime ({top_prob:.1%}). "
                f"Full cash/hedge overlay deployed. 21-day forward 95% CVaR is {tail_risk.cvar_95:.2%}."
            )
        elif pred_regime == "Risk-On":
            commentary = (
                f"Constructive Risk-On Stance: Low volatility and broad market breadth support "
                f"maximum equity beta ({top_prob:.1%}). Expected tail risk remains muted."
            )
        else:
            commentary = (
                f"Transitional / Late-Cycle Stance: Mixed indicators. Hedged allocation "
                f"recommended with tight monitoring of changepoint probability ({prediction.changepoint_probability:.1%})."
            )

        return InvestmentCommitteeBriefing(
            as_of_date=prediction.timestamp,
            current_regime=pred_regime,
            regime_confidence=top_prob,
            conformal_prediction_set=[r.value for r in prediction.conformal_prediction_set],
            epistemic_uncertainty=prediction.uncertainty.epistemic_uncertainty,
            aleatoric_uncertainty=prediction.uncertainty.aleatoric_uncertainty,
            changepoint_probability=prediction.changepoint_probability,
            recommended_equity_exposure=allocation_weights.get("equity_beta", 0.5),
            recommended_cash_hedge=allocation_weights.get("cash_hedge", 0.5),
            forward_var_95=tail_risk.var_95,
            forward_cvar_95=tail_risk.cvar_95,
            commentary=commentary,
        )
