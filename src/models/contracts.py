"""
Central Model Output Contracts & Typed Data Structures.

Every model in the engine emits predictions convertible to the canonical RegimePrediction contract.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class RegimeLabel(str, Enum):
    """The five canonical market regimes defined by Zetheta Algorithms."""
    RISK_ON = "Risk-On"
    RISK_OFF = "Risk-Off"
    TRANSITIONAL = "Transitional"
    LATE_CYCLE = "Late-Cycle"
    POST_SHOCK = "Post-Shock"


@dataclass(frozen=True)
class RegimeProbabilities:
    """Categorical probability distribution over the five canonical market regimes."""
    risk_on: float
    late_cycle: float
    transitional: float
    post_shock: float
    risk_off: float

    def to_array(self) -> np.ndarray:
        """Return probability vector in canonical order [Risk-On, Late-Cycle, Transitional, Post-Shock, Risk-Off]."""
        return np.array(
            [self.risk_on, self.late_cycle, self.transitional, self.post_shock, self.risk_off],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            RegimeLabel.RISK_ON.value: self.risk_on,
            RegimeLabel.LATE_CYCLE.value: self.late_cycle,
            RegimeLabel.TRANSITIONAL.value: self.transitional,
            RegimeLabel.POST_SHOCK.value: self.post_shock,
            RegimeLabel.RISK_OFF.value: self.risk_off,
        }

    def validate(self, tol: float = 1e-4) -> bool:
        """Validate probability axioms (non-negativity and unit sum)."""
        probs = self.to_array()
        if np.any(probs < -tol) or np.any(probs > 1.0 + tol):
            return False
        return abs(float(np.sum(probs)) - 1.0) <= tol

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RegimeProbabilities":
        """Construct from 5-element numpy array."""
        if len(arr) != 5:
            raise ValueError(f"Expected array of shape (5,), got shape {arr.shape}")
        # Normalize if slight numerical drift
        total = float(np.sum(arr))
        normalized = arr / total if total > 0 else arr
        return cls(
            risk_on=float(normalized[0]),
            late_cycle=float(normalized[1]),
            transitional=float(normalized[2]),
            post_shock=float(normalized[3]),
            risk_off=float(normalized[4]),
        )


@dataclass(frozen=True)
class UncertaintyMetrics:
    """Rigorous breakdown of uncertainty."""
    predictive_uncertainty: float   # Total predictive entropy H(p)
    epistemic_uncertainty: float    # Parameter ignorance / ensemble dispersion
    aleatoric_uncertainty: float    # Expected data noise / irreducible entropy

    def validate(self) -> bool:
        """Epistemic and aleatoric uncertainties must be non-negative."""
        return (
            self.predictive_uncertainty >= 0.0
            and self.epistemic_uncertainty >= -1e-6
            and self.aleatoric_uncertainty >= -1e-6
        )


@dataclass(frozen=True)
class ModelLineage:
    """Traceability metadata for audit and reproduction."""
    model_name: str
    model_version: str
    feature_snapshot_id: str
    data_snapshot_id: str
    training_period: str
    random_seed: int
    git_commit: Optional[str] = None


@dataclass(frozen=True)
class RegimePrediction:
    """The central unified prediction contract produced by every model and ensemble."""
    timestamp: datetime
    regime_probabilities: RegimeProbabilities
    predicted_regime: RegimeLabel
    uncertainty: UncertaintyMetrics
    conformal_prediction_set: List[RegimeLabel]
    changepoint_probability: float
    lineage: ModelLineage
    explanation_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.regime_probabilities.validate():
            raise ValueError(
                f"Invalid regime probabilities: {self.regime_probabilities.to_dict()}"
            )
        if not (0.0 <= self.changepoint_probability <= 1.0):
            raise ValueError(
                f"Changepoint probability must be in [0, 1], got {self.changepoint_probability}"
            )
        if not self.conformal_prediction_set:
            raise ValueError("Conformal prediction set cannot be empty.")
