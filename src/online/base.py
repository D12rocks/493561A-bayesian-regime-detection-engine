"""
Sequential & Online Inference Architecture.

Implements Sequential Monte Carlo (Bootstrap Particle Filter),
Bayesian Online Changepoint Detection (BOCPD), streaming Dirichlet updates,
and the Two-Speed Online/Batch Reconciliation diagnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.models.contracts import RegimeProbabilities


class BaseOnlineFilter(ABC):
    """
    Abstract interface for fast intraday state estimation.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset state tracking to prior distribution."""
        pass

    @abstractmethod
    def update(self, observation: np.ndarray) -> RegimeProbabilities:
        """
        Process single real-time observation vector and emit updated posterior regime probabilities.
        Execution budget: < 10 ms.
        """
        pass


class BaseChangepointDetector(ABC):
    """
    Abstract interface for Bayesian Online Changepoint Detection (BOCPD).
    """

    @abstractmethod
    def update(self, observation: float) -> Tuple[float, np.ndarray]:
        """
        Update run-length posterior given new observation.
        Returns:
            Tuple[float, np.ndarray]: (changepoint_probability, run_length_distribution)
        """
        pass


class TwoSpeedReconciler:
    """
    Diagnostic tool comparing the fast online particle state estimate against
    the nightly batch reference model posterior.
    Computes Kullback-Leibler (KL) Divergence and Total Variation Distance (TVD).
    """

    def __init__(self, max_allowable_kl: float = 0.25) -> None:
        self.max_allowable_kl = max_allowable_kl

    def reconcile(
        self,
        p_online: RegimeProbabilities,
        p_batch_ref: RegimeProbabilities,
    ) -> Dict[str, Any]:
        eps = 1e-12
        p_on = p_online.to_array() + eps
        p_ref = p_batch_ref.to_array() + eps

        # Normalize
        p_on = p_on / np.sum(p_on)
        p_ref = p_ref / np.sum(p_ref)

        # Forward KL: D_KL(P_online || P_batch) = sum p_on * log(p_on / p_ref)
        kl_div = float(np.sum(p_on * np.log(p_on / p_ref)))
        
        # Total Variation Distance: 0.5 * sum |p_on - p_ref|
        tvd = float(0.5 * np.sum(np.abs(p_on - p_ref)))
        
        is_consistent = kl_div <= self.max_allowable_kl

        return {
            "kl_divergence": kl_div,
            "total_variation_distance": tvd,
            "is_consistent": is_consistent,
            "max_threshold": self.max_allowable_kl,
            "requires_ad_hoc_batch_trigger": not is_consistent,
        }
