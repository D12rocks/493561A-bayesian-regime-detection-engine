"""
Abstract Base Class for all Regime Detection Models.

Enforces standardized inference API, diagnostic reporting, and lineage traceability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.contracts import (
    ModelLineage,
    RegimeLabel,
    RegimePrediction,
    RegimeProbabilities,
    UncertaintyMetrics,
)


class BaseRegimeModel(ABC):
    """
    Abstract base interface for all regime detection models in the engine.
    Ensures strict parity across Frequentist HMM, Bayesian HMM, RS-VAR, BNN, and Foundation Models.
    """

    def __init__(self, name: str, version: str = "1.0.0", random_seed: int = 42) -> None:
        self.name = name
        self.version = version
        self.random_seed = random_seed
        self.is_fitted = False
        self._feature_snapshot_id: str = "uninitialized"
        self._data_snapshot_id: str = "uninitialized"
        self._training_period: str = "uninitialized"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseRegimeModel":
        """
        Fit model parameters to historical feature matrix X.
        Must respect chronological ordering without future lookahead.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Produce out-of-sample posterior regime probabilities.
        Returns ndarray of shape (N, 5) strictly lying on the probability simplex.
        Columns must match [Risk-On, Late-Cycle, Transitional, Post-Shock, Risk-Off].
        """
        pass

    def predict_regime(self, X: pd.DataFrame) -> List[RegimeLabel]:
        """Classify observations into the maximum a posteriori (MAP) regime."""
        probs = self.predict_proba(X)
        canonical_labels = [
            RegimeLabel.RISK_ON,
            RegimeLabel.LATE_CYCLE,
            RegimeLabel.TRANSITIONAL,
            RegimeLabel.POST_SHOCK,
            RegimeLabel.RISK_OFF,
        ]
        map_indices = np.argmax(probs, axis=1)
        return [canonical_labels[idx] for idx in map_indices]

    @abstractmethod
    def diagnostics(self) -> Dict[str, Any]:
        """
        Return model-specific statistical diagnostics.
        For MCMC models: R-hat, ESS, divergences, BFMI.
        For frequentist models: log-likelihood, AIC/BIC, condition number.
        """
        pass

    def get_lineage(self) -> ModelLineage:
        """Return provenance metadata for the fitted model instance."""
        return ModelLineage(
            model_name=self.name,
            model_version=self.version,
            feature_snapshot_id=self._feature_snapshot_id,
            data_snapshot_id=self._data_snapshot_id,
            training_period=self._training_period,
            random_seed=self.random_seed,
        )

    def set_lineage(
        self,
        feature_snapshot_id: str,
        data_snapshot_id: str,
        training_period: str,
    ) -> None:
        """Record data and feature snapshot provenance."""
        self._feature_snapshot_id = feature_snapshot_id
        self._data_snapshot_id = data_snapshot_id
        self._training_period = training_period
