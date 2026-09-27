"""
Sequential Monte Carlo (Bootstrap Particle Filter) for Online Regime Tracking.

Implements REQ-050:
- Fast intraday Sequential Monte Carlo (Bootstrap Particle Filter)
- P=1,000 particles tracking regime states s_t in {0, 1, 2, 3, 4}
- Systematic resampling triggered when Effective Sample Size N_eff < P / 2
- Emits real-time posterior probabilities within < 5 ms
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import multivariate_normal

from src.online.base import BaseOnlineFilter
from src.models.contracts import RegimeProbabilities


class BootstrapParticleFilter(BaseOnlineFilter):
    """
    Bootstrap Particle Filter tracking discrete regime states across streaming market observations.
    """

    def __init__(
        self,
        n_particles: int = 1000,
        trans_mat: Optional[np.ndarray] = None,
        means: Optional[np.ndarray] = None,
        covs: Optional[np.ndarray] = None,
        random_seed: int = 42,
    ) -> None:
        self.n_particles = n_particles
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # 5-regime transition matrix (canonical order)
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = np.full((5, 5), 0.05)
            np.fill_diagonal(self.trans_mat, 0.80)
            self.trans_mat /= np.sum(self.trans_mat, axis=1, keepdims=True)

        self.means = means
        self.covs = covs

        # Particles state: s in {0, 1, 2, 3, 4}, weights w in R
        self.particles: np.ndarray = np.zeros(n_particles, dtype=int)
        self.weights: np.ndarray = np.ones(n_particles) / n_particles
        self.reset()

    def reset(self) -> None:
        """Reset particles to uniform prior across the 5 regimes."""
        self.particles = self.rng.choice(5, size=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _systematic_resample(self) -> None:
        """Systematic low-variance resampling algorithm."""
        n = self.n_particles
        positions = (self.rng.rand() + np.arange(n)) / n
        indexes = np.zeros(n, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights = np.ones(n) / n

    def update(self, observation: np.ndarray) -> RegimeProbabilities:
        """
        Process single real-time observation vector:
        1. Predict next state for each particle: sample s_t ~ A(s_{t-1}, :)
        2. Weight particle by observation likelihood: w_t = w_{t-1} * P(y_t | s_t)
        3. Normalize weights
        4. Resample if Effective Sample Size N_eff < P / 2
        """
        # 1. State transition
        for i in range(self.n_particles):
            curr_state = self.particles[i]
            trans_p = self.trans_mat[curr_state]
            self.particles[i] = self.rng.choice(5, p=trans_p)

        # 2. Likelihood weighting
        if self.means is not None and self.covs is not None:
            likelihoods = np.zeros(self.n_particles)
            for state in range(5):
                st_mask = self.particles == state
                if np.any(st_mask):
                    try:
                        p_val = multivariate_normal.pdf(
                            observation, mean=self.means[state], cov=self.covs[state], allow_singular=True
                        )
                    except Exception:
                        p_val = 1e-6
                    likelihoods[st_mask] = max(p_val, 1e-12)
            self.weights *= likelihoods
        else:
            # Fallback heuristic weighting based on return (feature 0)
            ret = observation[0]
            scores = np.array([
                np.exp(ret * 10),        # Risk-On
                np.exp(ret * 4),         # Late-Cycle
                np.exp(-abs(ret) * 10),  # Transitional
                np.exp(ret * 2),         # Post-Shock
                np.exp(-ret * 10),       # Risk-Off
            ])
            self.weights *= scores[self.particles]

        # Normalize
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # 3. Check Effective Sample Size
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2.0:
            self._systematic_resample()

        # 4. Compute empirical state probabilities
        regime_probs = np.zeros(5)
        for state in range(5):
            regime_probs[state] = np.sum(self.weights[self.particles == state])
            
        regime_probs /= np.maximum(np.sum(regime_probs), 1e-12)
        return RegimeProbabilities.from_array(regime_probs)
