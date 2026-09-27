"""
Regime-Conditioned Monte Carlo Simulation and Tail Risk Engine.

Implements REQ-093 and Specification Section Risk/Simulation:
- Simulates 10,000 forward market paths over 21-day horizons
- Transitions states via Markov transition dynamics P(S_{t+1} | S_t)
- Draws innovations from fat-tailed Student-t emissions
- Calculates 21-day 95% and 99% Value-at-Risk (VaR) and Expected Shortfall (CVaR)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import t as student_t

from src.models.contracts import RegimeProbabilities
from src.simulation.base import BaseRegimeSimulator, TailRiskMetrics


class RegimeConditionedSimulator(BaseRegimeSimulator):
    """
    Simulates forward return distributions conditioned on current regime probabilities
    and Markov transition matrix dynamics.
    """

    def __init__(
        self,
        trans_mat: Optional[np.ndarray] = None,
        regime_means: Optional[np.ndarray] = None,
        regime_vols: Optional[np.ndarray] = None,
        degrees_of_freedom: float = 5.0,
    ) -> None:
        # Default empirical transition matrix if not supplied
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = np.array([
                [0.85, 0.08, 0.05, 0.01, 0.01],  # Risk-On
                [0.05, 0.80, 0.10, 0.03, 0.02],  # Late-Cycle
                [0.10, 0.10, 0.70, 0.05, 0.05],  # Transitional
                [0.05, 0.05, 0.10, 0.75, 0.05],  # Post-Shock
                [0.02, 0.03, 0.05, 0.10, 0.80],  # Risk-Off
            ])

        # Regime annualized parameters [Risk-On, Late-Cycle, Transitional, Post-Shock, Risk-Off]
        self.regime_means = regime_means if regime_means is not None else np.array([
            0.18,   # Risk-On: +18% ann
            0.08,   # Late-Cycle: +8% ann
            0.02,   # Transitional: +2% ann
            0.14,   # Post-Shock: +14% ann rebound
            -0.25,  # Risk-Off: -25% ann crash
        ]) / 252.0

        self.regime_vols = regime_vols if regime_vols is not None else np.array([
            0.12,   # Risk-On: 12% vol
            0.18,   # Late-Cycle: 18% vol
            0.15,   # Transitional: 15% vol
            0.24,   # Post-Shock: 24% vol
            0.35,   # Risk-Off: 35% vol
        ]) / np.sqrt(252.0)

        self.df = degrees_of_freedom

    def simulate_paths(
        self,
        current_probs: RegimeProbabilities,
        horizon_days: int = 21,
        n_paths: int = 10000,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Simulates forward daily returns: shape (n_paths, horizon_days).
        """
        rng = np.random.RandomState(random_seed)
        p_init = current_probs.to_array()

        paths = np.zeros((n_paths, horizon_days))
        # Initial regime sampling
        current_states = rng.choice(5, size=n_paths, p=p_init)

        for d in range(horizon_days):
            # Transition states for day d > 0
            if d > 0:
                for s in range(5):
                    mask = current_states == s
                    if np.any(mask):
                        n_sub = int(np.sum(mask))
                        current_states[mask] = rng.choice(5, size=n_sub, p=self.trans_mat[s])

            # Draw fat-tailed Student-t returns
            means = self.regime_means[current_states]
            scales = self.regime_vols[current_states]
            
            # Standard Student-t scaled
            t_draws = student_t.rvs(df=self.df, size=n_paths, random_state=rng)
            # Rescale variance: Var(t) = df / (df - 2)
            std_t = np.sqrt(self.df / (self.df - 2.0))
            innovations = (t_draws / std_t) * scales
            paths[:, d] = means + innovations

        return paths

    def compute_tail_risk(
        self,
        simulated_returns: np.ndarray,
        horizon_days: int = 21,
    ) -> TailRiskMetrics:
        """
        Calculates cumulative horizon returns and tail risk quantiles.
        """
        # Cumulative return over horizon: prod(1 + r) - 1
        cum_returns = np.prod(1.0 + simulated_returns, axis=1) - 1.0

        var_95 = float(np.percentile(cum_returns, 5.0))
        var_99 = float(np.percentile(cum_returns, 1.0))

        cvar_95 = float(np.mean(cum_returns[cum_returns <= var_95]))
        cvar_99 = float(np.mean(cum_returns[cum_returns <= var_99]))

        mean_ret = float(np.mean(cum_returns))
        vol = float(np.std(cum_returns))

        return TailRiskMetrics(
            horizon_days=horizon_days,
            var_95=round(var_95, 4),
            cvar_95=round(cvar_95, 4),
            var_99=round(var_99, 4),
            cvar_99=round(cvar_99, 4),
            expected_return=round(mean_ret, 4),
            path_volatility=round(vol, 4),
        )
