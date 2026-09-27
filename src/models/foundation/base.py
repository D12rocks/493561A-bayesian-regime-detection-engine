"""
Unified Foundation Model Adapter Interface.

Allows seamless swapping of time-series foundation models (Chronos, TimesFM, Lag-Llama, Moirai)
without modifying downstream ensembling, calibration, or regime inference logic.
Includes explicit execution capability probes to handle hardware limitations without fabricating outputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.base import BaseRegimeModel


class FoundationModelAdapter(BaseRegimeModel):
    """
    Abstract adapter unifying time-series foundation models into the regime detection engine.
    Extracts probabilistic representations or zero-shot forecasts and maps them via a
    Bayesian classification head to the 5 canonical regimes.
    """

    def __init__(
        self,
        model_id: str,
        context_length: int = 64,
        prediction_horizon: int = 5,
        device: str = "auto",
        version: str = "1.0.0",
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=f"Foundation_{model_id}", version=version, random_seed=random_seed)
        self.model_id = model_id
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        self._is_hardware_supported: Optional[bool] = None
        self._limitation_notes: str = ""

    @abstractmethod
    def check_execution_capability(self) -> Tuple[bool, str]:
        """
        Verify whether the underlying foundation model can run on the host system.
        Returns:
            Tuple[bool, str]: (is_supported, detailed_diagnostic_message)
        """
        pass

    @abstractmethod
    def extract_representation(self, series: pd.Series) -> np.ndarray:
        """
        Extract latent temporal embedding or zero-shot distribution from the foundation backbone.
        Raises RuntimeError if check_execution_capability() is False. Never fabricates values.
        """
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FoundationModelAdapter":
        """Fit downstream Bayesian head onto foundation model representations."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Map foundation representations to 5-regime probabilities."""
        pass

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "context_length": self.context_length,
            "prediction_horizon": self.prediction_horizon,
            "hardware_supported": self._is_hardware_supported,
            "limitation_notes": self._limitation_notes,
        }
