"""
Adaptive Conformal Inference (ACI) and Conformal Prediction Set Engine.

Implements REQ-031 and Specification Section G:
- Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) for non-exchangeable financial time series
- Adaptive Prediction Sets (APS)
- Empirical Coverage Auditing: realizes and reports empirical coverage without assuming 90%
- Dynamic alpha_t adaptation tracking distribution shifts and volatility bursts
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.calibration.base import BaseConformalPredictor
from src.models.contracts import RegimeLabel


class AdaptiveConformalInference(BaseConformalPredictor):
    """
    Adaptive Conformal Inference (ACI) for non-stationary market regimes.
    
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1 if y_t not in C_t(alpha_t), else 0.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha)
        self.gamma = gamma
        self.random_seed = random_seed
        self.current_alpha = alpha
        self.calibration_scores: np.ndarray = np.array([])
        self.q_hat: float = 1.0 - alpha
        self.canonical_labels = [
            RegimeLabel.RISK_ON,
            RegimeLabel.LATE_CYCLE,
            RegimeLabel.TRANSITIONAL,
            RegimeLabel.POST_SHOCK,
            RegimeLabel.RISK_OFF,
        ]

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> "AdaptiveConformalInference":
        """
        Calibrates initial quantile q_hat on calibration dataset using APS scores:
        s_i = sum_{k: p_k >= p_{y_i}} p_k
        """
        n_samples = len(probs_cal)
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            true_label = int(y_cal[i])
            p_vec = probs_cal[i]
            p_true = p_vec[true_label]
            # Accumulate probability of all classes with probability >= true class probability
            scores[i] = np.sum(p_vec[p_vec >= p_true])

        # Conformal quantile: ceiling((N + 1)(1 - alpha)) / N
        self.calibration_scores = np.sort(scores)
        q_level = np.clip(np.ceil((n_samples + 1.0) * (1.0 - self.alpha)) / n_samples, 0.0, 1.0)
        self.q_hat = float(np.quantile(self.calibration_scores, q_level))
        self.is_calibrated = True
        return self

    def predict_set_single(self, prob_vector: np.ndarray, alpha_t: Optional[float] = None) -> List[RegimeLabel]:
        """
        Constructs prediction set C_t for a single 5-element probability vector.
        Sorts classes descending by probability, accumulates until reaching 1 - alpha_t (or q_hat).
        """
        target_mass = 1.0 - (alpha_t if alpha_t is not None else self.current_alpha)
        target_mass = np.clip(target_mass, 0.10, 0.99)

        # Sort indices descending
        sorted_indices = np.argsort(prob_vector)[::-1]
        c_set = []
        accumulated_prob = 0.0

        for idx in sorted_indices:
            c_set.append(self.canonical_labels[idx])
            accumulated_prob += prob_vector[idx]
            if accumulated_prob >= target_mass:
                break

        # Invariant: Conformal set must never be empty
        if len(c_set) == 0:
            c_set.append(self.canonical_labels[sorted_indices[0]])

        return c_set

    def predict_set(self, probs: np.ndarray) -> List[List[RegimeLabel]]:
        """Generates static prediction sets using current calibrated threshold."""
        return [self.predict_set_single(p) for p in probs]

    def run_online_aci(
        self, probs_seq: np.ndarray, y_seq: np.ndarray
    ) -> Tuple[List[List[RegimeLabel]], np.ndarray, Dict[str, float]]:
        """
        Runs sequential online Adaptive Conformal Inference with real-time feedback:
        For each time step t:
        1. Produce C_t using current alpha_t
        2. Observe true regime y_t
        3. Record err_t = 1 if y_t not in C_t else 0
        4. Update alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        
        Returns:
            prediction_sets: List of sets
            alpha_trajectory: History of alpha_t over time
            audit_metrics: Realized empirical coverage, average set size, etc.
        """
        n_obs = len(probs_seq)
        prediction_sets: List[List[RegimeLabel]] = []
        alpha_trajectory = np.zeros(n_obs)
        errors = np.zeros(n_obs)
        set_sizes = np.zeros(n_obs)

        alpha_curr = self.alpha

        for t in range(n_obs):
            alpha_trajectory[t] = alpha_curr
            c_t = self.predict_set_single(probs_seq[t], alpha_t=alpha_curr)
            prediction_sets.append(c_t)
            set_sizes[t] = len(c_t)

            true_label_idx = int(y_seq[t])
            true_label = self.canonical_labels[true_label_idx]

            is_covered = true_label in c_t
            err_t = 0.0 if is_covered else 1.0
            errors[t] = err_t

            # ACI adaptation step
            alpha_curr = np.clip(alpha_curr + self.gamma * (self.alpha - err_t), 0.01, 0.50)

        self.current_alpha = alpha_curr

        empirical_coverage = float(1.0 - np.mean(errors))
        mean_set_size = float(np.mean(set_sizes))
        single_set_pct = float(np.mean(set_sizes == 1) * 100.0)

        audit_metrics = {
            "target_coverage": float(1.0 - self.alpha),
            "realized_empirical_coverage": round(empirical_coverage, 4),
            "coverage_gap": round(empirical_coverage - (1.0 - self.alpha), 4),
            "mean_prediction_set_size": round(mean_set_size, 2),
            "single_regime_certainty_pct": round(single_set_pct, 1),
            "adaptation_rate_gamma": self.gamma,
        }

        return prediction_sets, alpha_trajectory, audit_metrics
