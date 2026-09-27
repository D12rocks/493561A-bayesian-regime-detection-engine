"""
Regime-Switching Vector Autoregression (RS-VAR) via Hamilton Filter.

Implements REQ-022: Multivariate Markov-Switching VAR with:
- State-dependent intercept c_k and autoregressive transition matrices Phi_k
- Hamilton (1989) forward filter for strictly point-in-time filtered regime probabilities
- Kim (1994) backward smoothing algorithm for full-sample inference
- Multivariate Gaussian conditional innovation densities
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from src.models.base import BaseRegimeModel
from src.models.contracts import RegimeLabel


class RegimeSwitchingVAR(BaseRegimeModel):
    """
    Multivariate Regime-Switching Vector Autoregression (RS-VAR) model.
    
    Y_t = c_{S_t} + sum_{p=1}^P Phi_{p, S_t} Y_{t-p} + epsilon_t,  epsilon_t ~ N(0, Sigma_{S_t})
    """

    def __init__(
        self,
        name: str = "rs_var",
        n_regimes: int = 5,
        lags: int = 1,
        n_iter: int = 60,
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=name, version="1.0.0", random_seed=random_seed)
        self.n_regimes = n_regimes
        self.lags = lags
        self.n_iter = n_iter
        
        # Model parameters
        self.trans_mat: np.ndarray = np.array([])          # (K, K)
        self.intercepts: np.ndarray = np.array([])         # (K, D)
        self.ar_matrices: np.ndarray = np.array([])        # (K, P, D, D)
        self.covariances: np.ndarray = np.array([])        # (K, D, D)
        self.state_to_regime_map: Dict[int, int] = {}
        self.feature_columns: List[str] = []
        self._log_likelihood: float = np.nan

    def _hamilton_filter(
        self,
        y: np.ndarray,
        trans_mat: np.ndarray,
        intercepts: np.ndarray,
        ar_matrices: np.ndarray,
        covariances: np.ndarray,
        init_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes the Hamilton forward filter:
        xi_{t|t-1} = A^T xi_{t-1|t-1}
        xi_{t|t} = (xi_{t|t-1} * eta_t) / sum(...)
        """
        n_obs, d = y.shape
        p = self.lags
        k = self.n_regimes
        
        t_effective = n_obs - p
        filtered_probs = np.zeros((t_effective, k))
        predicted_probs = np.zeros((t_effective, k))
        eta = np.zeros((t_effective, k))
        log_lik = 0.0

        xi_curr = init_probs.copy()

        for t in range(t_effective):
            # Observation at t+p
            y_curr = y[t + p]
            # Lagged observations: y_{t+p-1}, ..., y_t
            lags_flat = np.array([y[t + p - l] for l in range(1, p + 1)]) # (P, D)

            # Prediction step: xi_{t|t-1} = trans_mat^T * xi_{t-1|t-1}
            xi_pred = trans_mat.T @ xi_curr
            predicted_probs[t] = xi_pred

            # Innovation density for each regime: N(y_curr | c_k + sum Phi_k y_{t-l}, Sigma_k)
            for state in range(k):
                cond_mean = intercepts[state].copy()
                for l in range(p):
                    cond_mean += ar_matrices[state, l] @ lags_flat[l]

                try:
                    density = multivariate_normal.pdf(
                        y_curr, mean=cond_mean, cov=covariances[state], allow_singular=True
                    )
                except Exception:
                    density = 1e-6
                eta[t, state] = max(density, 1e-12)

            # Update step
            numerator = xi_pred * eta[t]
            f_t = np.sum(numerator)
            if f_t > 0:
                xi_curr = numerator / f_t
                log_lik += np.log(f_t)
            else:
                xi_curr = np.ones(k) / k

            filtered_probs[t] = xi_curr

        return filtered_probs, predicted_probs, log_lik

    def _kim_smoother(
        self,
        filtered_probs: np.ndarray,
        predicted_probs: np.ndarray,
        trans_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Kim (1994) backward smoothing:
        xi_{t|T} = xi_{t|t} * sum_j [ trans_mat_{t, j} * xi_{t+1|T} / predicted_probs_{t+1|t} ]
        """
        t_effective, k = filtered_probs.shape
        smoothed_probs = np.zeros((t_effective, k))
        smoothed_probs[-1] = filtered_probs[-1]

        for t in range(t_effective - 2, -1, -1):
            next_smooth = smoothed_probs[t + 1]
            next_pred = np.maximum(predicted_probs[t + 1], 1e-12)
            ratio = next_smooth / next_pred

            smoothed_probs[t] = filtered_probs[t] * (trans_mat @ ratio)
            sum_s = np.sum(smoothed_probs[t])
            if sum_s > 0:
                smoothed_probs[t] /= sum_s

        return smoothed_probs

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RegimeSwitchingVAR":
        self.feature_columns = list(X.columns)
        X_clean = X.dropna().values
        n_obs, d = X_clean.shape
        k = self.n_regimes
        p = self.lags
        rng = np.random.RandomState(self.random_seed)

        # Initialize parameters
        # Transition matrix with sticky diagonal
        self.trans_mat = np.full((k, k), 0.05)
        np.fill_diagonal(self.trans_mat, 0.8)
        self.trans_mat /= np.sum(self.trans_mat, axis=1, keepdims=True)

        # Intercepts: spread across data quantiles
        self.intercepts = np.zeros((k, d))
        quantiles = np.linspace(0.1, 0.9, k)
        for st in range(k):
            self.intercepts[st] = np.quantile(X_clean, quantiles[st], axis=0)

        # AR matrices initialized small
        self.ar_matrices = np.zeros((k, p, d, d))
        for st in range(k):
            for l in range(p):
                self.ar_matrices[st, l] = np.eye(d) * 0.1

        # Covariances initialized to sample covariance
        sample_cov = np.cov(X_clean, rowvar=False) + np.eye(d) * 0.05
        self.covariances = np.array([sample_cov.copy() for _ in range(k)])

        init_probs = np.ones(k) / k

        # EM loop over Hamilton filter and Kim smoother
        t_eff = n_obs - p
        for iteration in range(self.n_iter):
            # E-step: Hamilton filter + Kim smoother
            filtered, predicted, log_lik = self._hamilton_filter(
                X_clean, self.trans_mat, self.intercepts, self.ar_matrices, self.covariances, init_probs
            )
            smoothed = self._kim_smoother(filtered, predicted, self.trans_mat)
            self._log_likelihood = log_lik

            # M-step: Update transition matrix and VAR parameters using smoothed probabilities
            # 1. Update Transition Matrix
            new_trans = np.zeros((k, k))
            for t in range(t_eff - 1):
                numerator = np.outer(filtered[t], smoothed[t + 1] / np.maximum(predicted[t + 1], 1e-12)) * self.trans_mat
                new_trans += numerator
            row_sums = np.sum(new_trans, axis=1, keepdims=True)
            self.trans_mat = new_trans / np.maximum(row_sums, 1e-12)

            # 2. Update VAR intercepts and AR matrices via weighted OLS
            y_targets = X_clean[p:]  # (T_eff, D)
            lags_input = np.array([X_clean[p - l : n_obs - l] for l in range(1, p + 1)]) # (P, T_eff, D)
            
            for st in range(k):
                weights = smoothed[:, st]  # (T_eff,)
                weight_sum = np.sum(weights)
                if weight_sum < 1e-6:
                    continue

                w_sqrt = np.sqrt(weights)[:, None]
                # Design matrix: [1, y_{t-1}, ..., y_{t-p}]
                x_design_blocks = [np.ones((t_eff, 1))]
                for l in range(p):
                    x_design_blocks.append(lags_input[l])
                x_design = np.column_stack(x_design_blocks)  # (T_eff, 1 + P*D)

                # Weighted least squares
                x_w = x_design * w_sqrt
                y_w = y_targets * w_sqrt
                try:
                    beta, _, _, _ = np.linalg.lstsq(x_w, y_w, rcond=1e-4)
                    self.intercepts[st] = beta[0]
                    for l in range(p):
                        self.ar_matrices[st, l] = beta[1 + l * d : 1 + (l + 1) * d].T

                    # Residual covariance
                    preds = x_design @ beta
                    resids = y_targets - preds
                    weighted_resids = resids * w_sqrt
                    self.covariances[st] = (weighted_resids.T @ weighted_resids) / weight_sum + np.eye(d) * 1e-4
                except Exception:
                    pass

        self.is_fitted = True

        # Map states to canonical regimes based on intercept of feature 0 (return)
        f0_means = self.intercepts[:, 0]
        sorted_indices = np.argsort(f0_means) # ascending: worst (Risk-Off) to best (Risk-On)
        
        self.state_to_regime_map = {
            int(sorted_indices[4]): 0,  # Risk-On
            int(sorted_indices[3]): 1,  # Late-Cycle
            int(sorted_indices[2]): 2,  # Transitional
            int(sorted_indices[1]): 3,  # Post-Shock
            int(sorted_indices[0]): 4,  # Risk-Off
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")

        cols = self.feature_columns if self.feature_columns else X.columns
        X_sub = X[cols].ffill().fillna(0.0).values
        n_obs = len(X_sub)
        p = self.lags

        # Hamilton filter runs forward strictly point-in-time
        init_probs = np.ones(self.n_regimes) / self.n_regimes
        filtered, _, _ = self._hamilton_filter(
            X_sub, self.trans_mat, self.intercepts, self.ar_matrices, self.covariances, init_probs
        )

        # Pad initial lag rows with equal probability
        padded_filtered = np.vstack([
            np.full((p, self.n_regimes), 1.0 / self.n_regimes),
            filtered,
        ])

        # Map to canonical 5 regimes
        canon_probs = np.zeros((n_obs, 5))
        for raw_st, canon_idx in self.state_to_regime_map.items():
            if canon_idx < 5:
                canon_probs[:, canon_idx] += padded_filtered[:, raw_st]

        row_sums = np.sum(canon_probs, axis=1, keepdims=True)
        return canon_probs / np.maximum(row_sums, 1e-12)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "log_likelihood": self._log_likelihood,
            "n_regimes": self.n_regimes,
            "ar_lags": self.lags,
            "transition_matrix": self.trans_mat.tolist() if self.is_fitted else [],
            "intercepts": self.intercepts.tolist() if self.is_fitted else [],
        }
