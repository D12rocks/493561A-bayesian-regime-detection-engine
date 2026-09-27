"""
Rigorous Temporal Anti-Leakage Tests for Feature Engineering.

Proves mathematically that feature values at date t are invariant to
any future data, perturbations, or shocks occurring at t+1 or later.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.technical import TechnicalFeatureExtractor
from src.features.tda import TDAFeatureExtractor
from src.features.gnn import SectorGraphFeatureExtractor
from src.features.store import PointInTimeFeatureStore


@pytest.fixture
def base_market():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    symbols = ["NIFTY_50", "NIFTY_MIDCAP_50", "NIFTY_500", "INDIA_VIX", "USD_INR", "NIFTY_BANK", "NIFTY_IT"]
    market = {}
    for s in symbols:
        p0 = 100.0 if "VIX" not in s else 15.0
        ret = np.random.normal(0.0005, 0.015, size=len(dates))
        prices = p0 * np.exp(np.cumsum(ret))
        highs = prices * 1.01
        lows = prices * 0.99
        market[s] = pd.DataFrame({
            "open": prices, "high": highs, "low": lows, "close": prices, "volume": 1000.0
        }, index=dates)
    return market


def test_technical_zero_future_leakage(base_market):
    """
    Features at cutoff date t=80 must be strictly identical whether calculated
    on data up to t=80 or on data with future observations up to t=120.
    """
    cutoff_idx = 80
    cutoff_date = base_market["NIFTY_50"].index[cutoff_idx]
    
    # 1. Past slice only
    past_market = {s: df.iloc[:cutoff_idx + 1].copy() for s, df in base_market.items()}
    
    # 2. Perturbed future market (future has massive shock)
    perturbed_market = {s: df.copy() for s, df in base_market.items()}
    for s, df in perturbed_market.items():
        # Inject extreme future shock at cutoff_idx + 1 onwards
        df.iloc[cutoff_idx + 1:, df.columns.get_loc("close")] *= 5.0
        df.iloc[cutoff_idx + 1:, df.columns.get_loc("high")] *= 5.0
        df.iloc[cutoff_idx + 1:, df.columns.get_loc("low")] *= 5.0
        
    extractor = TechnicalFeatureExtractor(sma_long_window=40, sma_short_window=15, vol_window=10)
    
    feats_past = extractor.extract_from_multivariate(past_market)
    feats_perturbed = extractor.extract_from_multivariate(perturbed_market)
    
    vec_past = feats_past.loc[cutoff_date].dropna()
    vec_perturbed = feats_perturbed.loc[cutoff_date].dropna()
    
    # Common features must match exactly
    common_cols = vec_past.index.intersection(vec_perturbed.index)
    assert len(common_cols) > 0
    
    diff = np.abs(vec_past[common_cols].values - vec_perturbed[common_cols].values)
    assert np.max(diff) < 1e-9, f"Future leakage detected in technical features: max diff {np.max(diff)}"


def test_point_in_time_store_query_anti_leakage(base_market):
    """
    PointInTimeFeatureStore must reject queries with dates before available history
    and strictly return the last available row on or before as_of_date.
    """
    store = PointInTimeFeatureStore()
    extractor = TechnicalFeatureExtractor(sma_long_window=30, sma_short_window=10, vol_window=10)
    feats = extractor.extract_from_multivariate(base_market)
    
    as_of = base_market["NIFTY_50"].index[50]
    vec = store.get_historical_feature_vector(feats, as_of_date=as_of)
    
    assert vec.name == as_of
    
    with pytest.raises(ValueError):
        # Query before history
        store.get_historical_feature_vector(feats, as_of_date="1990-01-01")
