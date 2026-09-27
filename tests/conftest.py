"""
Pytest Fixtures & Reusable Synthetic Market Data Generators.

Provides deterministic synthetic market data with known regime transitions
for unit, statistical, and leakage tests.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pytest

from src.models.contracts import (
    ModelLineage,
    RegimeLabel,
    RegimePrediction,
    RegimeProbabilities,
    UncertaintyMetrics,
)


@pytest.fixture
def synthetic_market_data() -> pd.DataFrame:
    """
    Generate 500 business days of synthetic multi-asset Indian market data.
    Assets: NIFTY_50, INDIA_VIX, USD_INR, GILT_10Y.
    """
    np.random.seed(42)
    n_days = 500
    start_date = datetime(2020, 1, 1)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    # Simulate realistic geometric Brownian motion with regime shifts
    returns_nifty = np.random.normal(0.0005, 0.012, n_days)
    # Inject shock
    returns_nifty[100:130] = np.random.normal(-0.008, 0.035, 30)

    price_nifty = 12000.0 * np.exp(np.cumsum(returns_nifty))
    vix = 14.0 + np.maximum(0, -returns_nifty * 500) + np.random.normal(0, 1.0, n_days)
    vix = np.clip(vix, 9.0, 75.0)

    usdinr = 72.0 + np.cumsum(np.random.normal(0.01, 0.15, n_days))
    gilt = 6.5 + np.cumsum(np.random.normal(0.00, 0.02, n_days))

    df = pd.DataFrame(
        {
            "NIFTY_50": price_nifty,
            "INDIA_VIX": vix,
            "USD_INR": usdinr,
            "GILT_10Y": gilt,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def valid_regime_probabilities() -> RegimeProbabilities:
    return RegimeProbabilities(
        risk_on=0.60,
        late_cycle=0.15,
        transitional=0.10,
        post_shock=0.05,
        risk_off=0.10,
    )


@pytest.fixture
def valid_regime_prediction(valid_regime_probabilities: RegimeProbabilities) -> RegimePrediction:
    lineage = ModelLineage(
        model_name="TestEnsemble",
        model_version="1.0.0",
        feature_snapshot_id="feat_snap_test_001",
        data_snapshot_id="data_snap_test_001",
        training_period="2009-2017",
        random_seed=42,
    )
    uncertainty = UncertaintyMetrics(
        predictive_uncertainty=1.12,
        epistemic_uncertainty=0.18,
        aleatoric_uncertainty=0.94,
    )
    return RegimePrediction(
        timestamp=datetime(2023, 6, 15, 15, 30),
        regime_probabilities=valid_regime_probabilities,
        predicted_regime=RegimeLabel.RISK_ON,
        uncertainty=uncertainty,
        conformal_prediction_set=[RegimeLabel.RISK_ON, RegimeLabel.LATE_CYCLE],
        changepoint_probability=0.04,
        lineage=lineage,
        explanation_metadata={"notes": "Synthetic test prediction"},
    )
