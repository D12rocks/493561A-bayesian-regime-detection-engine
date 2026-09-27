"""
Conviction-Aware Dynamic Asset Allocation Overlay.

Implements REQ-090 and Specification Section I:
- Regime-conditioned equity targets scaled by model conviction
- Conformal set width penalty (wider set -> lower conviction -> smaller tilts)
- Epistemic uncertainty throttling
- No-trade hysteresis band minimizing wasteful portfolio turnover
- Mutual Fund scheme constraints (Equity bound in [20%, 100%], Cash in [0%, 80%])
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.backtest.base import BaseAllocationOverlay


class ConvictionAwareAllocationOverlay(BaseAllocationOverlay):
    """
    Transparent, deterministic tactical allocation overlay mapping regime probabilities,
    conformal set sizes, and uncertainty into disciplined portfolio tilts.
    """

    def __init__(
        self,
        base_equity_weight: float = 0.65,
        no_trade_band: float = 0.05,
        max_daily_turnover: float = 0.15,
        min_equity: float = 0.20,
        max_equity: float = 1.00,
    ) -> None:
        self.base_equity_weight = base_equity_weight
        self.no_trade_band = no_trade_band
        self.max_daily_turnover = max_daily_turnover
        self.min_equity = min_equity
        self.max_equity = max_equity

        # Strategic regime unconstrained equity targets
        # [0: Risk-On, 1: Late-Cycle, 2: Transitional, 3: Post-Shock, 4: Risk-Off]
        self.regime_equity_targets = np.array([0.95, 0.70, 0.50, 0.80, 0.25])

    def compute_conviction(
        self,
        probabilities: np.ndarray,
        conformal_set_sizes: Optional[np.ndarray] = None,
        epistemic_uncertainties: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculates conviction score C_t in [0, 1]:
        C_t = max_k(p_{t,k}) * (1 - (|C_t| - 1) / 4) * exp(-epistemic_t)
        """
        n_samples = len(probabilities)
        max_p = np.max(probabilities, axis=1)

        # Set size penalty (1 to 5 regimes -> factor in [1.0, 0.0])
        if conformal_set_sizes is not None:
            size_factor = np.clip(1.0 - (conformal_set_sizes - 1.0) / 4.0, 0.1, 1.0)
        else:
            size_factor = np.ones(n_samples)

        # Epistemic penalty
        if epistemic_uncertainties is not None:
            epistemic_factor = np.exp(-np.clip(epistemic_uncertainties, 0.0, 3.0))
        else:
            epistemic_factor = np.ones(n_samples)

        conviction = max_p * size_factor * epistemic_factor
        return np.clip(conviction, 0.0, 1.0)

    def compute_weights(
        self,
        probabilities: np.ndarray,
        date_index: pd.DatetimeIndex,
        conformal_set_sizes: Optional[np.ndarray] = None,
        epistemic_uncertainties: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Computes point-in-time portfolio weights (Equity vs Cash/G-Sec)
        incorporating hysteresis bands and turnover limits.
        """
        n_samples = len(probabilities)
        convictions = self.compute_conviction(
            probabilities, conformal_set_sizes, epistemic_uncertainties
        )

        # Raw expected regime target equity
        # w*_t = sum_k p_{t, k} * w_k
        raw_regime_targets = probabilities @ self.regime_equity_targets

        # Scale tilt from benchmark by conviction
        # target_t = base + conviction * (raw - base)
        target_equity_series = self.base_equity_weight + convictions * (
            raw_regime_targets - self.base_equity_weight
        )
        target_equity_series = np.clip(target_equity_series, self.min_equity, self.max_equity)

        # Apply Hysteresis & Turnover Controls sequentially
        actual_equity = np.zeros(n_samples)
        actual_equity[0] = target_equity_series[0]

        for t in range(1, n_samples):
            prev_w = actual_equity[t - 1]
            desired_w = target_equity_series[t]
            delta = desired_w - prev_w

            # 1. No-trade hysteresis band
            if abs(delta) < self.no_trade_band:
                actual_equity[t] = prev_w
            else:
                # 2. Maximum turnover cap
                clipped_delta = np.clip(delta, -self.max_daily_turnover, self.max_daily_turnover)
                actual_equity[t] = prev_w + clipped_delta

        weights_df = pd.DataFrame(
            {
                "equity_weight": actual_equity,
                "cash_weight": 1.0 - actual_equity,
                "conviction": convictions,
                "target_equity_raw": target_equity_series,
            },
            index=date_index,
        )
        return weights_df
