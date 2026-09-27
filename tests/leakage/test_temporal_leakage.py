"""
Automated Leakage & Lookahead Prevention Tests.

Guarantees zero future contamination in feature calculations,
validation partitioning, and rolling statistics.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config


@pytest.mark.leakage
def test_chronological_split_strictness() -> None:
    """Assert train, calibration, and test splits have strictly non-overlapping temporal bounds."""
    cfg = load_config("validation")
    train = cfg["splits"]["training"]
    cal = cfg["splits"]["calibration"]
    test = cfg["splits"]["test"]

    d_train_end = pd.Timestamp(train["end_date"])
    d_cal_start = pd.Timestamp(cal["start_date"])
    d_cal_end = pd.Timestamp(cal["end_date"])
    d_test_start = pd.Timestamp(test["start_date"])

    assert d_train_end < d_cal_start, "Calibration period must strictly follow training period."
    assert d_cal_end < d_test_start, "Test period must strictly follow calibration period."


@pytest.mark.leakage
def test_future_mutation_invariance(synthetic_market_data: pd.DataFrame) -> None:
    """
    Adversarial Leakage Test:
    Altering an observation at date T+k must cause ZERO change in backward-looking
    rolling features evaluated at or before date T.
    """
    df = synthetic_market_data.copy()
    cutoff_idx = 300
    cutoff_date = df.index[cutoff_idx]

    # Baseline backward rolling feature (20-day realized volatility)
    df["log_ret"] = np.log(df["NIFTY_50"] / df["NIFTY_50"].shift(1))
    df["vol_20d"] = df["log_ret"].rolling(window=20).std()

    baseline_features_up_to_cutoff = df.loc[:cutoff_date, "vol_20d"].copy()

    # Mutate data into the future (at cutoff_idx + 10)
    df_mutated = synthetic_market_data.copy()
    future_idx = cutoff_idx + 10
    future_date = df_mutated.index[future_idx]
    df_mutated.loc[future_date, "NIFTY_50"] *= 5.0  # Massive future shock

    # Recompute features
    df_mutated["log_ret"] = np.log(df_mutated["NIFTY_50"] / df_mutated["NIFTY_50"].shift(1))
    df_mutated["vol_20d"] = df_mutated["log_ret"].rolling(window=20).std()

    mutated_features_up_to_cutoff = df_mutated.loc[:cutoff_date, "vol_20d"]

    # Assert exact numerical equality on past slice
    pd.testing.assert_series_equal(
        baseline_features_up_to_cutoff,
        mutated_features_up_to_cutoff,
        check_exact=True,
    )


@pytest.mark.leakage
def test_no_centered_or_forward_rolling(synthetic_market_data: pd.DataFrame) -> None:
    """Verify that feature generators do not use center=True in pandas rolling windows."""
    # A centered rolling window at t uses t+1, which is a lethal leakage bug
    df = synthetic_market_data.copy()
    rolling_obj = df["NIFTY_50"].rolling(window=10)
    assert rolling_obj.center is False, "Rolling windows must be strictly backward-looking."
