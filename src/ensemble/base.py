"""
Base Ensemble Architecture & Aggregation Engines.

Implements Bayesian Model Averaging (BMA), Constrained Stacking on the Simplex,
and formal decomposition into Epistemic and Aleatoric Uncertainty.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.base import BaseRegimeModel
from src.models.contracts import (
    ModelLineage,
    RegimeLabel,
    RegimeProbabilities,
    UncertaintyMetrics,
)


class BaseEnsemble(ABC):
    """
    Abstract Base Class for Multi-Model Regime Ensembling.
    Enforces quality hurdles before allowing candidate models into the active federation.
    """

    def __init__(self, name: str, max_allowed_ece: float = 0.12) -> None:
        self.name = name
        self.max_allowed_ece = max_allowed_ece
        self.models: List[BaseRegimeModel] = []
        self.weights: Optional[np.ndarray] = None
        self.model_metadata: List[Dict[str, Any]] = []

    def register_model(
        self,
        model: BaseRegimeModel,
        validation_ece: float,
        diagnostics: Dict[str, Any],
    ) -> None:
        """
        Register a candidate model into the ensemble pool with strict admission hurdles.
        """
        if validation_ece > self.max_allowed_ece:
            raise ValueError(
                f"Model '{model.name}' rejected: ECE {validation_ece:.4f} exceeds max threshold {self.max_allowed_ece:.4f}"
            )
        self.models.append(model)
        self.model_metadata.append(
            {
                "name": model.name,
                "version": model.version,
                "validation_ece": validation_ece,
                "diagnostics": diagnostics,
            }
        )

    @abstractmethod
    def fit_weights(self, X_cal: pd.DataFrame, y_cal: np.ndarray) -> np.ndarray:
        """
        Compute aggregation weights using calibration data.
        Must enforce weights lie on probability simplex (non-negative, sum to 1).
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute ensemble probability predictions:
        P_ens(X) = sum_m w_m * P_m(X)
        """
        if self.weights is None or len(self.models) == 0:
            raise RuntimeError("Ensemble must be fitted with registered models before prediction.")
        
        # Shape: (M, N, 5)
        model_predictions = np.array([m.predict_proba(X) for m in self.models])
        
        # Weighted combination: sum over M -> (N, 5)
        ensemble_probs = np.tensordot(self.weights, model_predictions, axes=(0, 0))
        return ensemble_probs

    def compute_uncertainty(self, X: pd.DataFrame) -> List[UncertaintyMetrics]:
        """
        Decompose total predictive uncertainty into Epistemic and Aleatoric components.
        - Total Entropy: H(E[p])
        - Aleatoric Uncertainty: E[H(p)] (average entropy across models)
        - Epistemic Uncertainty: Total Entropy - Aleatoric Uncertainty (Mutual Information)
        """
        eps = 1e-12
        # (M, N, 5)
        all_probs = np.array([m.predict_proba(X) for m in self.models])
        M, N, K = all_probs.shape
        
        # Model entropy per sample: H_m(x) = -sum_k p_{mk} log(p_{mk})
        per_model_entropy = -np.sum(all_probs * np.log(all_probs + eps), axis=2) # (M, N)
        
        # Aleatoric: weighted average of entropies
        aleatoric = np.tensordot(self.weights, per_model_entropy, axes=(0, 0)) # (N,)
        
        # Expected probability: E[p] = sum_m w_m * p_m
        ensemble_probs = self.predict_proba(X) # (N, 5)
        total_entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + eps), axis=1) # (N,)
        
        # Epistemic = Total - Aleatoric
        epistemic = np.maximum(0.0, total_entropy - aleatoric)
        
        return [
            UncertaintyMetrics(
                predictive_uncertainty=float(total_entropy[i]),
                epistemic_uncertainty=float(epistemic[i]),
                aleatoric_uncertainty=float(aleatoric[i]),
            )
            for i in range(N)
        ]
