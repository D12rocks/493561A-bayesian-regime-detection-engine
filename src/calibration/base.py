"""
Calibration & Conformal Prediction Engine.

Implements Adaptive Prediction Sets (APS), Adaptive Conformal Inference (ACI)
for non-exchangeable financial time-series, Expected Calibration Error (ECE),
and multi-class Brier scores.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.models.contracts import RegimeLabel


class BaseConformalPredictor(ABC):
    """
    Abstract interface for conformal prediction sets.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        """
        alpha: target significance level (e.g. 0.10 -> 90% coverage).
        """
        self.alpha = alpha
        self.is_calibrated = False

    @abstractmethod
    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> "BaseConformalPredictor":
        """Calibrate non-conformity threshold on validation/calibration dataset."""
        pass

    @abstractmethod
    def predict_set(self, probs: np.ndarray) -> List[List[RegimeLabel]]:
        """Generate prediction set C(x) for each observation."""
        pass


class BaseCalibrator(ABC):
    """
    Abstract interface for probability calibrators (e.g. Temperature Scaling, Dirichlet Calibration).
    """

    @abstractmethod
    def fit(self, logits_or_probs: np.ndarray, y: np.ndarray) -> "BaseCalibrator":
        pass

    @abstractmethod
    def calibrate(self, logits_or_probs: np.ndarray) -> np.ndarray:
        pass
