"""
Technical, Volatility, and Cross-Asset Breadth Feature Engineering.

Implements REQ-015: Backward-looking rolling technical, cross-asset,
and market breadth indicators computed strictly point-in-time without future leakage.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.features.base import BaseFeatureTransformer


class TechnicalFeatureExtractor(BaseFeatureTransformer):
    """
    Computes financial time-series features strictly using historical observations <= t.
    
    Features engineered:
    - Log returns (1d, 5d, 21d)
    - Realized Volatility: 21-day EWMA (RiskMetrics lambda=0.94) and 21-day standard deviation
    - Parkinson High-Low Volatility (21-day)
    - Trend: Distance to 200-day and 50-day Simple Moving Averages
    - Relative Strength Index (RSI-14)
    - Volatility dynamics: INDIA_VIX level and 5-day rate of change
    - Currency dynamics: USD/INR 21-day rolling log return
    - Market Breadth Spreads:
        * Midcap vs NIFTY 50 21-day return spread
        * NIFTY 500 vs NIFTY 50 21-day return spread
        * Sector divergence: Bank Nifty vs IT Nifty 21-day return spread
    """

    def __init__(
        self,
        name: str = "technical_features",
        ewma_lambda: float = 0.94,
        rsi_window: int = 14,
        sma_long_window: int = 200,
        sma_short_window: int = 50,
        vol_window: int = 21,
    ) -> None:
        super().__init__(name=name)
        self.ewma_lambda = ewma_lambda
        self.rsi_window = rsi_window
        self.sma_long_window = sma_long_window
        self.sma_short_window = sma_short_window
        self.vol_window = vol_window

    def fit(self, df: pd.DataFrame) -> "TechnicalFeatureExtractor":
        # Technical transformations are deterministic functions of backward rolling windows;
        # no lookahead parameters to fit.
        self.is_fitted = True
        return self

    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _compute_ewma_volatility(self, returns: pd.Series, decay: float = 0.94) -> pd.Series:
        """Annualized EWMA volatility using RiskMetrics exponential decay."""
        # var_t = decay * var_{t-1} + (1 - decay) * ret_t^2
        var_series = (returns ** 2).ewm(alpha=1.0 - decay, adjust=False).mean()
        annualized_vol = np.sqrt(np.maximum(var_series, 1e-12) * 252.0)
        return annualized_vol

    def _compute_parkinson_volatility(
        self, high: pd.Series, low: pd.Series, window: int = 21
    ) -> pd.Series:
        """
        Parkinson extreme-value volatility estimator:
        sigma_P = sqrt( 252 / (window * 4 * ln(2)) * sum( (ln(H/L))^2 ) )
        """
        log_hl_ratio_sq = (np.log(np.maximum(high, 1e-6) / np.maximum(low, 1e-6))) ** 2
        rolling_sum = log_hl_ratio_sq.rolling(window=window, min_periods=window).sum()
        scale = 252.0 / (window * 4.0 * np.log(2.0))
        parkinson_vol = np.sqrt(np.maximum(rolling_sum * scale, 0.0))
        return parkinson_vol

    def extract_from_multivariate(
        self,
        prices: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Extract features from dictionary of individual asset OHLCV dataframes.
        
        Args:
            prices: Dictionary mapping symbol string to DataFrame with DatetimeIndex
                    and columns ['open', 'high', 'low', 'close', 'volume'].
        
        Returns:
            DataFrame with DatetimeIndex and engineered technical feature columns.
        """
        if "NIFTY_50" not in prices:
            raise ValueError("NIFTY_50 data is mandatory for technical feature extraction.")

        nifty = prices["NIFTY_50"].sort_index()
        idx = nifty.index

        feats = pd.DataFrame(index=idx)

        # 1. NIFTY 50 Log Returns
        log_ret_1d = np.log(nifty["close"] / nifty["close"].shift(1))
        feats["nifty_ret_1d"] = log_ret_1d
        feats["nifty_ret_5d"] = np.log(nifty["close"] / nifty["close"].shift(5))
        feats["nifty_ret_21d"] = np.log(nifty["close"] / nifty["close"].shift(21))

        # 2. Realized & EWMA Volatilities
        feats["nifty_vol_ewma_21d"] = self._compute_ewma_volatility(log_ret_1d, self.ewma_lambda)
        feats["nifty_vol_realized_21d"] = (
            log_ret_1d.rolling(window=self.vol_window, min_periods=self.vol_window).std()
            * np.sqrt(252.0)
        )

        # 3. Parkinson Volatility
        if "high" in nifty.columns and "low" in nifty.columns:
            feats["nifty_vol_parkinson_21d"] = self._compute_parkinson_volatility(
                nifty["high"], nifty["low"], window=self.vol_window
            )

        # 4. Moving Average Trend Distances
        sma_200 = nifty["close"].rolling(window=self.sma_long_window, min_periods=self.sma_long_window).mean()
        sma_50 = nifty["close"].rolling(window=self.sma_short_window, min_periods=self.sma_short_window).mean()
        feats["nifty_dist_sma200"] = (nifty["close"] / sma_200) - 1.0
        feats["nifty_dist_sma50"] = (nifty["close"] / sma_50) - 1.0

        # 5. Momentum: RSI-14
        feats["nifty_rsi_14"] = self._compute_rsi(nifty["close"], window=self.rsi_window)

        # 6. Volatility Regime: INDIA VIX
        if "INDIA_VIX" in prices:
            vix = prices["INDIA_VIX"]["close"].reindex(idx).ffill()
            feats["vix_level"] = vix
            feats["vix_ret_5d"] = (vix - vix.shift(5)) / np.maximum(vix.shift(5), 1e-4)

        # 7. Currency Dynamics: USD/INR
        if "USD_INR" in prices:
            usdinr = prices["USD_INR"]["close"].reindex(idx).ffill()
            feats["usdinr_ret_21d"] = np.log(usdinr / usdinr.shift(21))

        # 8. Breadth: Midcap vs Largecap
        if "NIFTY_MIDCAP_50" in prices:
            midcap = prices["NIFTY_MIDCAP_50"]["close"].reindex(idx).ffill()
            midcap_ret_21d = np.log(midcap / midcap.shift(21))
            feats["breadth_midcap_ret_21d"] = midcap_ret_21d - feats["nifty_ret_21d"]
            feats["breadth_midcap_ratio"] = np.log(midcap / nifty["close"])

        # 9. Breadth: NIFTY 500 Broad Participation
        if "NIFTY_500" in prices:
            nifty500 = prices["NIFTY_500"]["close"].reindex(idx).ffill()
            n500_ret_21d = np.log(nifty500 / nifty500.shift(21))
            feats["breadth_broad_ret_21d"] = n500_ret_21d - feats["nifty_ret_21d"]

        # 10. Sector Divergence: Bank vs IT
        if "NIFTY_BANK" in prices and "NIFTY_IT" in prices:
            bank = prices["NIFTY_BANK"]["close"].reindex(idx).ffill()
            it = prices["NIFTY_IT"]["close"].reindex(idx).ffill()
            bank_ret_21d = np.log(bank / bank.shift(21))
            it_ret_21d = np.log(it / it.shift(21))
            feats["sector_bank_it_divergence_21d"] = bank_ret_21d - it_ret_21d

        return feats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback transform if single price dataframe passed with 'close', 'high', 'low'.
        """
        return self.extract_from_multivariate({"NIFTY_50": df})
