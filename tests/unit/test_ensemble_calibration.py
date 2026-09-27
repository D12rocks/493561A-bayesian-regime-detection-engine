"""
Unit Tests for Ensembling, Conformal Calibration, and Baseline Benchmarking.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import FrequentistHMM, BayesianHMM
from src.ensemble import BayesianModelAveraging, ConstrainedStackingEnsemble
from src.calibration import AdaptiveConformalInference, TemperatureScalingCalibrator
from src.evaluation import (
    ClimatologyBaseline,
    PersistenceBaseline,
    BenchmarkTournament,
    compute_log_loss,
    compute_ranked_probability_score,
)


@pytest.fixture
def sample_ensemble_data():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    
    df = pd.DataFrame({
        "feat1": np.random.normal(0, 1, 100),
        "feat2": np.random.normal(0, 1, 100),
    }, index=dates)

    # 5 ground truth labels
    y = np.random.choice(5, size=100)
    return df, y


def test_constrained_stacking_simplex_weights(sample_ensemble_data):
    df, y = sample_ensemble_data
    
    # Fit two models
    m1 = FrequentistHMM(n_regimes=5, n_iter=10, random_seed=42).fit(df)
    m2 = FrequentistHMM(n_regimes=5, n_iter=10, random_seed=101).fit(df)

    stacker = ConstrainedStackingEnsemble()
    stacker.register_model(m1, {"log_loss": 1.2}, m1.diagnostics())
    stacker.register_model(m2, {"log_loss": 1.3}, m2.diagnostics())

    weights = stacker.fit_weights(df, y)
    
    # Simplex check: non-negative and sums to 1.0
    assert len(weights) == 2
    assert (weights >= 0.0).all()
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)

    probs = stacker.predict_proba(df)
    assert probs.shape == (100, 5)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_adaptive_conformal_inference(sample_ensemble_data):
    df, y = sample_ensemble_data
    
    # Generate mock probabilities
    np.random.seed(42)
    raw_p = np.random.exponential(scale=1.0, size=(100, 5))
    probs = raw_p / np.sum(raw_p, axis=1, keepdims=True)

    aci = AdaptiveConformalInference(alpha=0.10, gamma=0.02)
    
    # Calibrate on first 50
    aci.calibrate(probs[:50], y[:50])
    assert aci.is_calibrated

    # Run online ACI on next 50
    sets, alpha_traj, audit = aci.run_online_aci(probs[50:], y[50:])
    
    assert len(sets) == 50
    assert len(alpha_traj) == 50
    assert "realized_empirical_coverage" in audit
    assert "mean_prediction_set_size" in audit
    # Sets must never be empty
    assert all(len(s) > 0 for s in sets)


def test_temperature_scaling_calibrator():
    np.random.seed(42)
    raw_p = np.random.exponential(scale=1.0, size=(80, 5))
    probs = raw_p / np.sum(raw_p, axis=1, keepdims=True)
    y = np.random.choice(5, size=80)

    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(probs, y)
    assert calibrator.is_fitted
    assert calibrator.temperature > 0.0

    cal_p = calibrator.calibrate(probs)
    assert cal_p.shape == (80, 5)
    assert np.allclose(np.sum(cal_p, axis=1), 1.0, atol=1e-5)

    ece, mce, rel_df = TemperatureScalingCalibrator.compute_ece(cal_p, y, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(rel_df) == 5


def test_benchmark_baselines_and_tournament():
    np.random.seed(42)
    y = np.array([0, 0, 0, 1, 1, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1])
    n = len(y)

    clim = ClimatologyBaseline().fit(y)
    p_clim = clim.predict_proba(n)
    assert p_clim.shape == (n, 5)
    assert np.allclose(np.sum(p_clim, axis=1), 1.0)

    persist = PersistenceBaseline()
    p_persist = persist.predict_proba(y)
    assert p_persist.shape == (n, 5)
    assert np.allclose(np.sum(p_persist, axis=1), 1.0)

    # Tournament
    model_preds = {
        "Mock Model A": p_persist,
    }
    tourn_df = BenchmarkTournament.evaluate_models(model_preds, y)
    assert len(tourn_df) >= 3
    assert "ranked_prob_score_rps" in tourn_df.columns
    assert "skill_vs_persistence" in tourn_df.columns
