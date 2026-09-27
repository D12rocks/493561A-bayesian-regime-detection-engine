"""
Unit Tests for Conviction-Aware Allocation, Walk-Forward Backtester, Monte Carlo, and Scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import ConvictionAwareAllocationOverlay, WalkForwardBacktester
from src.simulation import RegimeConditionedSimulator, ScenarioEngine
from src.models.contracts import RegimeProbabilities


@pytest.fixture
def sample_market_returns():
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=300, freq="B")
    
    eq_ret = np.random.normal(0.0006, 0.012, size=len(dates))
    cash_ret = np.full(len(dates), 0.065 / 252.0)
    
    asset_df = pd.DataFrame({
        "equity_return": eq_ret,
        "cash_return": cash_ret,
    }, index=dates)

    bench_s = pd.Series(eq_ret, index=dates)

    # 5-regime mock probabilities
    raw_p = np.random.uniform(0.1, 0.9, size=(len(dates), 5))
    raw_p /= np.sum(raw_p, axis=1, keepdims=True)

    return dates, asset_df, bench_s, raw_p


def test_conviction_aware_allocation_overlay(sample_market_returns):
    dates, asset_df, bench_s, probs = sample_market_returns
    overlay = ConvictionAwareAllocationOverlay(base_equity_weight=0.65, no_trade_band=0.03)

    weights_df = overlay.compute_weights(probs, dates)

    assert "equity_weight" in weights_df.columns
    assert "cash_weight" in weights_df.columns
    assert "conviction" in weights_df.columns

    # Weights must sum to 1.0 and obey boundaries
    assert np.allclose(weights_df["equity_weight"] + weights_df["cash_weight"], 1.0)
    assert (weights_df["equity_weight"] >= 0.20).all()
    assert (weights_df["equity_weight"] <= 1.00).all()


def test_walk_forward_backtester(sample_market_returns):
    dates, asset_df, bench_s, probs = sample_market_returns
    overlay = ConvictionAwareAllocationOverlay()
    weights_df = overlay.compute_weights(probs, dates)

    backtester = WalkForwardBacktester(risk_free_rate=0.065, n_trials_tried=10)
    metrics, sim_df = backtester.run(weights_df, asset_df, bench_s, transaction_cost_bps=15.0)

    assert metrics.cagr is not None
    assert metrics.annualized_volatility > 0.0
    assert metrics.sharpe_ratio is not None
    assert metrics.max_drawdown <= 0.0
    assert metrics.turnover_annual >= 0.0

    # Deflated Sharpe Ratio
    dsr = backtester._compute_deflated_sharpe(sim_df["strategy_return"].values, metrics.sharpe_ratio)
    assert 0.0 <= dsr <= 1.0

    # Conviction tier evaluation
    tiers_df = backtester.evaluate_conviction_tiers(sim_df, weights_df["conviction"])
    assert len(tiers_df) > 0


def test_regime_conditioned_monte_carlo():
    sim = RegimeConditionedSimulator()
    p_risk_on = RegimeProbabilities(0.8, 0.1, 0.05, 0.03, 0.02)
    p_risk_off = RegimeProbabilities(0.02, 0.03, 0.05, 0.1, 0.8)

    paths_on = sim.simulate_paths(p_risk_on, horizon_days=21, n_paths=2000, random_seed=42)
    paths_off = sim.simulate_paths(p_risk_off, horizon_days=21, n_paths=2000, random_seed=42)

    risk_on_metrics = sim.compute_tail_risk(paths_on)
    risk_off_metrics = sim.compute_tail_risk(paths_off)

    # Risk-Off must have significantly more severe VaR and CVaR than Risk-On
    assert risk_off_metrics.var_99 < risk_on_metrics.var_99
    assert risk_off_metrics.cvar_99 < risk_on_metrics.cvar_99


def test_scenario_engine():
    engine = ScenarioEngine()
    
    # Historical definitions exist
    assert "covid_crash_2020" in engine.HISTORICAL_SCENARIOS
    assert "taper_tantrum_2013" in engine.HISTORICAL_SCENARIOS

    # Synthetic shock
    base_feat = pd.Series({"nifty_ret_1d": 0.001, "vix_level": 15.0, "breadth_midcap_ret_21d": 0.005})
    shocked = engine.inject_synthetic_shock(base_feat, "rbi_hawkish_surprise")

    # In hawkish surprise, Nifty return drops and VIX rises
    assert shocked["nifty_ret_1d"] < base_feat["nifty_ret_1d"]
    assert shocked["vix_level"] > base_feat["vix_level"]
