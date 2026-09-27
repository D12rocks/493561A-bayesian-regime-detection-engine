"""
Backtesting & Dynamic Allocation Overlay Engine.

Evaluates regime-directed asset allocation strategies over the 2019-2024 out-of-sample period,
incorporating transaction friction, slippage, and regime-conditioned drawdown attribution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    """Summary statistics for quantitative backtesting performance."""
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float
    tracking_error: float
    turnover_annual: float
    total_transaction_costs: float
    win_rate_monthly: float
    regime_drawdowns: Dict[str, float]


class BaseAllocationOverlay(ABC):
    """
    Abstract interface for mapping posterior regime probabilities into target asset weights.
    """

    @abstractmethod
    def compute_weights(
        self,
        probabilities: np.ndarray,
        date_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        probabilities: shape (N, 5)
        Returns DataFrame of asset weights of shape (N, num_assets) with sum(weights) <= 1.0.
        """
        pass


class BaseBacktester(ABC):
    """
    Abstract backtesting execution engine.
    """

    @abstractmethod
    def run(
        self,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        transaction_cost_bps: float = 15.0,
    ) -> BacktestMetrics:
        """Execute portfolio walk-forward backtest and compute performance attribution."""
        pass
