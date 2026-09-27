"""
Temperature Scaling and Expected Calibration Error (ECE) Engine.

Implements REQ-032:
- Post-hoc temperature scaling: p_cal = Softmax(z / T)
- Multi-class Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagram binned confidence vs accuracy
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.calibration.base import BaseCalibrator


class TemperatureScalingCalibrator(BaseCalibrator):
    """
    Optimizes a single temperature parameter T > 0 on validation set
    to minimize negative log-likelihood (cross-entropy).
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
        return np.log(clipped)

    def _softmax(self, logits: np.ndarray, temp: float) -> np.ndarray:
        scaled = logits / max(temp, 1e-4)
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TemperatureScalingCalibrator":
        """
        Finds T > 0 minimizing cross-entropy on validation predictions.
        """
        logits = self._probs_to_logits(probs)
        n_samples = len(y)

        # One-hot targets
        y_onehot = np.zeros((n_samples, probs.shape[1]))
        for i, val in enumerate(y):
            y_onehot[i, int(val)] = 1.0

        def nll_objective(temp: float) -> float:
            cal_probs = self._softmax(logits, temp)
            nll = -np.mean(np.sum(y_onehot * np.log(np.maximum(cal_probs, 1e-12)), axis=1))
            return float(nll)

        res = minimize_scalar(nll_objective, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return probs
        logits = self._probs_to_logits(probs)
        return self._softmax(logits, self.temperature)

    @staticmethod
    def compute_ece(
        probs: np.ndarray, y: np.ndarray, n_bins: int = 10
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        
        Returns:
            Tuple: (ece, mce, reliability_df)
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y).astype(float)
        n_total = len(y)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_records = []
        ece = 0.0
        mce = 0.0

        for b_idx in range(n_bins):
            b_low = bins[b_idx]
            b_high = bins[b_idx + 1]
            in_bin = (confidences > b_low) & (confidences <= b_high)
            n_in_bin = int(np.sum(in_bin))

            if n_in_bin > 0:
                bin_acc = float(np.mean(accuracies[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                abs_diff = abs(bin_acc - bin_conf)
                weight = n_in_bin / n_total

                ece += weight * abs_diff
                mce = max(mce, abs_diff)

                bin_records.append({
                    "bin_lower": round(b_low, 2),
                    "bin_upper": round(b_high, 2),
                    "count": n_in_bin,
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                    "calibration_gap": round(abs_diff, 4),
                })
            else:
                bin_records.append({
                    "bin_lower": round(b_low, 2),
                    "bin_upper": round(b_high, 2),
                    "count": 0,
                    "accuracy": np.nan,
                    "confidence": np.nan,
                    "calibration_gap": 0.0,
                })

        return float(ece), float(mce), pd.DataFrame(bin_records)
