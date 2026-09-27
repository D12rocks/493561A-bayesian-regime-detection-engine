"""
Bayesian Online Changepoint Detection (BOCPD) Engine.

Implements REQ-051:
- Exact Adams & MacKay (2007) Bayesian Online Changepoint Detection
- Recursive run-length posterior distribution P(r_t | x_{1:t})
- Online Student-t predictive densities under conjugate Normal-Inverse-Gamma priors
- Real-time changepoint probability P(r_t = 0 | x_{1:t}) for early regime shift detection
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import t as student_t

from src.online.base import BaseChangepointDetector


class BayesianOnlineChangepointDetector(BaseChangepointDetector):
    """
    Bayesian Online Changepoint Detection (BOCPD) via recursive message passing.
    """

    def __init__(
        self,
        hazard_rate: float = 100.0,
        mu_0: float = 0.0,
        kappa_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 0.0001,
        max_run_length: int = 500,
    ) -> None:
        self.hazard_rate = hazard_rate
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.max_run_length = max_run_length

        # Run length distribution: R[r] = P(r_t = r | x_{1:t})
        self.r_dist = np.array([1.0])

        # Sufficient statistics vectors for each potential run length r in [0, max_r]
        self.mu_t = np.array([mu_0])
        self.kappa_t = np.array([kappa_0])
        self.alpha_t = np.array([alpha_0])
        self.beta_t = np.array([beta_0])
        self.t_step = 0

    def reset(self) -> None:
        self.r_dist = np.array([1.0])
        self.mu_t = np.array([self.mu_0])
        self.kappa_t = np.array([self.kappa_0])
        self.alpha_t = np.array([self.alpha_0])
        self.beta_t = np.array([self.beta_0])
        self.t_step = 0

    def update(self, observation: float) -> Tuple[float, np.ndarray]:
        """
        Process single streaming observation x_t and update run-length distribution.
        
        Returns:
            Tuple[float, np.ndarray]: (changepoint_probability, run_length_probabilities)
        """
        self.t_step += 1
        x = float(observation)
        k_len = len(self.r_dist)

        # 1. Evaluate predictive probability under current Student-t parameters
        # Deg of freedom = 2 * alpha_t
        # Location = mu_t
        # Scale = sqrt(beta_t * (kappa_t + 1) / (alpha_t * kappa_t))
        df = 2.0 * self.alpha_t
        loc = self.mu_t
        scale = np.sqrt(np.maximum(self.beta_t * (self.kappa_t + 1.0) / (self.alpha_t * self.kappa_t), 1e-12))
        
        pred_probs = student_t.pdf(x, df=df, loc=loc, scale=scale)
        pred_probs = np.maximum(pred_probs, 1e-12)

        # 2. Hazard rate
        hazard = 1.0 / self.hazard_rate

        # 3. Growth probabilities: r_t = r_{t-1} + 1
        growth_probs = self.r_dist * pred_probs * (1.0 - hazard)

        # 4. Changepoint probability: r_t = 0
        cp_prob_joint = np.sum(self.r_dist * pred_probs * hazard)

        # 5. Form new joint distribution
        new_joint = np.zeros(k_len + 1)
        new_joint[0] = cp_prob_joint
        new_joint[1:] = growth_probs

        # 6. Normalize to obtain posterior P(r_t | x_{1:t})
        evidence = np.sum(new_joint)
        if evidence > 0:
            self.r_dist = new_joint / evidence
        else:
            self.r_dist = np.zeros(k_len + 1)
            self.r_dist[0] = 1.0

        # Truncate if exceeding max_run_length
        if len(self.r_dist) > self.max_run_length:
            self.r_dist = self.r_dist[:self.max_run_length]
            self.r_dist /= np.sum(self.r_dist)

        # 7. Update conjugate sufficient statistics
        # r = 0 resets to prior
        new_mu = np.zeros(len(self.r_dist))
        new_kappa = np.zeros(len(self.r_dist))
        new_alpha = np.zeros(len(self.r_dist))
        new_beta = np.zeros(len(self.r_dist))

        new_mu[0] = self.mu_0
        new_kappa[0] = self.kappa_0
        new_alpha[0] = self.alpha_0
        new_beta[0] = self.beta_0

        # r > 0 updates with observation x
        n_up = len(self.r_dist) - 1
        old_k = self.kappa_t[:n_up]
        old_mu = self.mu_t[:n_up]
        old_a = self.alpha_t[:n_up]
        old_b = self.beta_t[:n_up]

        k_plus = old_k + 1.0
        mu_plus = (old_k * old_mu + x) / k_plus
        a_plus = old_a + 0.5
        b_plus = old_b + 0.5 * (old_k / k_plus) * ((x - old_mu) ** 2)

        new_mu[1:] = mu_plus
        new_kappa[1:] = k_plus
        new_alpha[1:] = a_plus
        new_beta[1:] = b_plus

        self.mu_t = new_mu
        self.kappa_t = new_kappa
        self.alpha_t = new_alpha
        self.beta_t = new_beta

        changepoint_prob = float(self.r_dist[0])
        return changepoint_prob, self.r_dist
