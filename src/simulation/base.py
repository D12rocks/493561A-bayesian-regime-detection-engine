"""
Monte Carlo Regime-Conditioned Simulation & Tail Risk Engine.

Generates forward return paths conditioned on time-varying regime probabilities
and transition dynamics; computes forward Value at Risk (VaR) and Expected Shortfall (CVaR).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from src.models.contracts import RegimeProbabilities


@dataclass(frozen=True)
class TailRiskMetrics:
    """Tail risk and Value at Risk summary."""
    horizon_days: int
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    expected_return: float
    path_volatility: float


class BaseRegimeSimulator(ABC):
    """
    Abstract interface for regime-conditioned Monte Carlo path generation.
    """

    @abstractmethod
    def simulate_paths(
        self,
        current_probs: RegimeProbabilities,
        horizon_days: int = 21,
        n_paths: int = 10000,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate asset returns across future horizon.
        Returns ndarray of shape (n_paths, horizon_days).
        """
        pass

    @abstractmethod
    def compute_tail_risk(
        self,
        simulated_returns: np.ndarray,
        horizon_days: int = 21,
    ) -> TailRiskMetrics:
        """Calculate VaR and CVaR at 95% and 99% levels from simulated paths."""
        pass
