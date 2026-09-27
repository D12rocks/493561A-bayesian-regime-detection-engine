"""
Unit tests for Technical, TDA, and Sector Graph feature extractors.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.technical import TechnicalFeatureExtractor
from src.features.tda import TDAFeatureExtractor
from src.features.gnn import SectorGraphFeatureExtractor
from src.features.selection import FeatureAuditor


@pytest.fixture
def synthetic_market_data():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    
    symbols = ["NIFTY_50", "NIFTY_MIDCAP_50", "NIFTY_500", "INDIA_VIX", "USD_INR", "NIFTY_BANK", "NIFTY_IT"]
    market = {}
    
    for s in symbols:
        p0 = 100.0 if "VIX" not in s else 15.0
        ret = np.random.normal(0.0005, 0.015, size=len(dates))
        prices = p0 * np.exp(np.cumsum(ret))
        highs = prices * (1.0 + np.random.uniform(0.002, 0.01, size=len(dates)))
        lows = prices * (1.0 - np.random.uniform(0.002, 0.01, size=len(dates)))
        
        market[s] = pd.DataFrame({
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": 100000.0,
        }, index=dates)
        
    return market


def test_technical_feature_extraction(synthetic_market_data):
    extractor = TechnicalFeatureExtractor(sma_long_window=50, sma_short_window=20, vol_window=10)
    feats = extractor.extract_from_multivariate(synthetic_market_data)
    
    assert "nifty_ret_1d" in feats.columns
    assert "nifty_vol_ewma_21d" in feats.columns
    assert "nifty_vol_parkinson_21d" in feats.columns
    assert "nifty_dist_sma50" in feats.columns
    assert "nifty_rsi_14" in feats.columns
    assert "vix_level" in feats.columns
    assert "breadth_midcap_ret_21d" in feats.columns
    assert "sector_bank_it_divergence_21d" in feats.columns
    
    # RSI must be strictly bounded in [0, 100]
    valid_rsi = feats["nifty_rsi_14"].dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()
    
    # Volatility must be non-negative
    valid_vol = feats["nifty_vol_ewma_21d"].dropna()
    assert (valid_vol >= 0.0).all()


def test_tda_feature_extraction():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    returns_df = pd.DataFrame(
        np.random.normal(0, 0.02, size=(100, 4)),
        index=dates,
        columns=["NIFTY_50", "NIFTY_MIDCAP_50", "INDIA_VIX", "USD_INR"],
    )
    
    tda_extractor = TDAFeatureExtractor(window_size=30)
    tda_feats = tda_extractor.transform(returns_df)
    
    assert "tda_persistence_entropy_h0" in tda_feats.columns
    assert "tda_wasserstein_amplitude_h0" in tda_feats.columns
    assert "tda_max_persistence_h1" in tda_feats.columns
    assert "tda_landscape_norm_h1" in tda_feats.columns
    
    valid_entropy = tda_feats["tda_persistence_entropy_h0"].dropna()
    assert len(valid_entropy) > 0
    assert (valid_entropy >= 0.0).all()


def test_gnn_feature_extraction(synthetic_market_data):
    gnn_extractor = SectorGraphFeatureExtractor(window_size=30, embedding_dim=4)
    gnn_feats = gnn_extractor.extract_from_sector_prices(synthetic_market_data)
    
    assert "gnn_spectral_radius" in gnn_feats.columns
    assert "gnn_algebraic_connectivity" in gnn_feats.columns
    assert "gnn_graph_entropy" in gnn_feats.columns
    assert "gnn_bank_centrality" in gnn_feats.columns
    assert "gnn_embed_0" in gnn_feats.columns
    assert "gnn_embed_3" in gnn_feats.columns
    
    valid_spec = gnn_feats["gnn_spectral_radius"].dropna()
    assert (valid_spec >= 0.0).all()


def test_feature_auditor_stationarity():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    # Stationary series (white noise)
    stationary_series = np.random.normal(0, 1, size=200)
    # Non-stationary series (random walk)
    random_walk = np.cumsum(np.random.normal(0, 1, size=200))
    
    df = pd.DataFrame({
        "stat_feat": stationary_series,
        "non_stat_feat": random_walk,
    }, index=dates)
    
    audit_res = FeatureAuditor.test_stationarity(df)
    stat_row = audit_res[audit_res["feature"] == "stat_feat"].iloc[0]
    non_stat_row = audit_res[audit_res["feature"] == "non_stat_feat"].iloc[0]
    
    assert stat_row["is_stationary"] == True
    assert non_stat_row["is_stationary"] == False
