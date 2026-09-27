"""
Bayesian Model Averaging (BMA) and Constrained Stacking on the Simplex.

Implements REQ-030 and Specification Section G:
- Bayesian Model Averaging with marginal likelihood / AIC-BIC posterior model weights
- Constrained Stacking via Simplex Optimization (SLSQP):
  min_w - sum_t log(sum_m w_m * p_{m,t}(y_t))  s.t. sum w_m = 1, w_m >= 0
- Uncertainty decomposition: Predictive entropy, epistemic uncertainty (mutual information), aleatoric uncertainty
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.ensemble.base import BaseEnsemble
from src.models.base import BaseRegimeModel


class BayesianModelAveraging(BaseEnsemble):
    """
    Bayesian Model Averaging (BMA) combining heterogeneous regime models.
    Weights are derived from BIC/AIC marginal likelihood approximations:
    w_m = exp(-0.5 * delta_BIC_m) / sum(...)
    """

    def __init__(self, name: str = "bma_ensemble") -> None:
        super().__init__(name=name)

    def fit_weights(self, X_cal: pd.DataFrame, y_cal: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes BMA weights based on registered model diagnostics.
        If BIC is available, uses BIC posterior odds; otherwise uses uniform weights.
        """
        if len(self.models) == 0:
            raise RuntimeError("No models registered in BMA ensemble.")

        m_count = len(self.models)
        bics = []

        for meta in self.model_metadata:
            diag = meta.get("diagnostics", {})
            bic_val = diag.get("bic", None)
            if bic_val is not None and not np.isnan(bic_val):
                bics.append(float(bic_val))
            else:
                # Fallback to pseudo-BIC based on validation loss or default
                bics.append(0.0)

        bics = np.array(bics)
        # Shift relative to minimum BIC for numerical stability
        delta_bic = bics - np.min(bics)
        # Posterior model odds: P(M_m | D) proportional to exp(-0.5 * delta_BIC)
        weights = np.exp(-0.5 * np.clip(delta_bic, 0.0, 50.0))
        sum_w = np.sum(weights)
        if sum_w > 0:
            self.weights = weights / sum_w
        else:
            self.weights = np.ones(m_count) / m_count

        return self.weights


class ConstrainedStackingEnsemble(BaseEnsemble):
    """
    Constrained Stacking optimizer finding the non-negative combination on the simplex
    that minimizes cross-entropy / proper log-loss on calibration data.
    """

    def __init__(self, name: str = "stacking_ensemble") -> None:
        super().__init__(name=name)

    def fit_weights(self, X_cal: pd.DataFrame, y_cal: np.ndarray) -> np.ndarray:
        """
        Finds optimal weights w in Delta^(M-1) minimizing:
        L(w) = - (1/N) sum_{i=1}^N log( sum_{m=1}^M w_m * p_{m, i, y_i} )
        """
        if len(self.models) == 0:
            raise RuntimeError("No models registered in Stacking ensemble.")

        m_count = len(self.models)
        # Shape: (M, N, 5)
        all_preds = np.array([m.predict_proba(X_cal) for m in self.models])
        m_dim, n_samples, k_dim = all_preds.shape

        # One-hot target matrix if y_cal is 1D class labels
        if y_cal.ndim == 1:
            y_onehot = np.zeros((n_samples, 5))
            for i, label in enumerate(y_cal):
                y_onehot[i, int(label)] = 1.0
        else:
            y_onehot = y_cal

        def objective(w: np.ndarray) -> float:
            # w shape: (M,)
            # ens_prob: (N, 5)
            ens_prob = np.tensordot(w, all_preds, axes=(0, 0))
            # Cross-entropy
            ce = -np.mean(np.sum(y_onehot * np.log(np.maximum(ens_prob, 1e-12)), axis=1))
            return float(ce)

        # Simplex constraints: sum(w) = 1, w_m >= 0
        w0 = np.ones(m_count) / m_count
        bounds = [(0.0, 1.0) for _ in range(m_count)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        if res.success:
            self.weights = np.clip(res.x, 0.0, 1.0)
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(m_count) / m_count

        return self.weights
