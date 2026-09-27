"""
Frequentist Hidden Markov Model (HMM) for Market Regime Detection.

Implements REQ-020: 5-state Hidden Markov Model with Gaussian emissions,
EM Baum-Welch training, AIC/BIC model selection, Viterbi decoding,
and automatic semantic regime alignment.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm

from src.models.base import BaseRegimeModel
from src.models.contracts import RegimeLabel


class FrequentistHMM(BaseRegimeModel):
    """
    Frequentist 5-state Gaussian HMM trained via Baum-Welch Expectation-Maximization.
    
    States are automatically aligned to canonical regimes based on mean return and volatility:
    - Index 0: Risk-On (High positive return, low/moderate vol)
    - Index 1: Late-Cycle (Positive return, elevated vol)
    - Index 2: Transitional (Near-zero return, neutral vol)
    - Index 3: Post-Shock (High vol, mean-reverting bounce)
    - Index 4: Risk-Off (Negative return, extreme vol)
    """

    def __init__(
        self,
        name: str = "frequentist_hmm",
        n_regimes: int = 5,
        n_iter: int = 150,
        covariance_type: str = "full",
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=name, version="1.0.0", random_seed=random_seed)
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.model: Optional[hmm.GaussianHMM] = None
        self.state_to_regime_map: Dict[int, int] = {}
        self.feature_columns: List[str] = []
        self._aic: float = np.nan
        self._bic: float = np.nan
        self._log_likelihood: float = np.nan

    def _determine_regime_mapping(self, means: np.ndarray, covars: np.ndarray) -> Dict[int, int]:
        """
        Maps raw unsupervised cluster indices to the canonical 5 regimes
        [Risk-On: 0, Late-Cycle: 1, Transitional: 2, Post-Shock: 3, Risk-Off: 4]
        based on financial properties (return in feature 0, volatility/variance).
        """
        # Assume feature 0 is return, feature 1 is volatility/VIX
        ret_means = means[:, 0]
        vols = np.sqrt(covars[:, 0, 0]) if self.covariance_type == "full" else np.sqrt(covars[:, 0])

        # Score states: higher return is more bullish, lower vol is more quiet
        # Sort states from most bearish to most bullish
        # Risk-Off has lowest return and high vol
        # Risk-On has highest return and low vol
        sharpe_proxy = ret_means / (vols + 1e-6)
        sorted_by_sharpe = np.argsort(sharpe_proxy) # ascending: worst (Risk-Off) to best (Risk-On)

        # Canonical order: [0: Risk-On, 1: Late-Cycle, 2: Transitional, 3: Post-Shock, 4: Risk-Off]
        # Best Sharpe -> Risk-On (0)
        # 2nd best -> Late-Cycle (1)
        # 3rd -> Transitional (2)
        # 4th -> Post-Shock (3)
        # Worst Sharpe -> Risk-Off (4)
        mapping = {
            int(sorted_by_sharpe[4]): 0,  # Risk-On
            int(sorted_by_sharpe[3]): 1,  # Late-Cycle
            int(sorted_by_sharpe[2]): 2,  # Transitional
            int(sorted_by_sharpe[1]): 3,  # Post-Shock
            int(sorted_by_sharpe[0]): 4,  # Risk-Off
        }
        return mapping

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FrequentistHMM":
        self.feature_columns = list(X.columns)
        X_clean = X.dropna().values
        n_samples, n_features = X_clean.shape

        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_seed,
            min_covar=1e-3,
        )

        self.model.fit(X_clean)
        self.is_fitted = True

        # Calculate Log-Likelihood, AIC, BIC
        self._log_likelihood = float(self.model.score(X_clean))
        
        # Number of free parameters:
        # Initial state probs: K - 1
        # Transition matrix: K * (K - 1)
        # Means: K * D
        # Full Covariances: K * D * (D + 1) / 2
        k = self.n_regimes
        d = n_features
        cov_params = k * d * (d + 1) // 2 if self.covariance_type == "full" else k * d
        n_params = (k - 1) + k * (k - 1) + (k * d) + cov_params
        
        self._aic = -2.0 * self._log_likelihood + 2.0 * n_params
        self._bic = -2.0 * self._log_likelihood + n_params * np.log(n_samples)

        # Semantic alignment
        self.state_to_regime_map = self._determine_regime_mapping(
            self.model.means_, self.model.covars_
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")

        # Align columns
        cols = self.feature_columns if self.feature_columns else X.columns
        X_sub = X[cols].ffill().fillna(0.0).values

        raw_posteriors = self.model.predict_proba(X_sub)
        n_samples = len(X_sub)
        canonical_probs = np.zeros((n_samples, 5), dtype=np.float64)

        for raw_state, canon_idx in self.state_to_regime_map.items():
            if canon_idx < 5:
                canonical_probs[:, canon_idx] += raw_posteriors[:, raw_state]

        # Normalize across rows to ensure exact unit sum
        row_sums = np.sum(canonical_probs, axis=1, keepdims=True)
        canonical_probs = canonical_probs / np.maximum(row_sums, 1e-12)
        return canonical_probs

    def transition_matrix(self) -> np.ndarray:
        """Returns 5x5 transition matrix aligned to canonical regimes."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted.")
        raw_trans = self.model.transmat_
        canon_trans = np.zeros((5, 5), dtype=np.float64)
        for raw_i, canon_i in self.state_to_regime_map.items():
            for raw_j, canon_j in self.state_to_regime_map.items():
                if canon_i < 5 and canon_j < 5:
                    canon_trans[canon_i, canon_j] = raw_trans[raw_i, raw_j]
        # Normalize rows
        row_sums = np.sum(canon_trans, axis=1, keepdims=True)
        return canon_trans / np.maximum(row_sums, 1e-12)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "log_likelihood": self._log_likelihood,
            "aic": self._aic,
            "bic": self._bic,
            "n_regimes": self.n_regimes,
            "converged": self.model.monitor_.converged if self.model else False,
            "n_iter_done": self.model.monitor_.iter if self.model else 0,
            "transition_matrix": self.transition_matrix().tolist() if self.is_fitted else [],
        }
