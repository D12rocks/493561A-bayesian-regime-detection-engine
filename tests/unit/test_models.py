"""
Unit and Statistical Invariant Tests for the 5-Model Regime Lab.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    FrequentistHMM,
    BayesianHMM,
    RegimeSwitchingVAR,
    BayesianDeepLearningModel,
    ChronosRegimeAdapter,
    RegimeDurationAnalyzer,
)


@pytest.fixture
def sample_feature_data():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=180, freq="B")
    
    # 4 distinct features: return, vol, trend distance, VIX
    ret = np.random.normal(0.0005, 0.015, size=len(dates))
    vol = np.abs(np.random.normal(0.18, 0.05, size=len(dates)))
    trend = np.sin(np.linspace(0, 3 * np.pi, len(dates))) * 0.05
    vix = np.clip(np.random.normal(16.0, 4.0, size=len(dates)), 10.0, 45.0)

    df = pd.DataFrame({
        "nifty_ret_1d": ret,
        "nifty_vol_ewma_21d": vol,
        "nifty_dist_sma50": trend,
        "vix_level": vix,
    }, index=dates)
    return df


def test_frequentist_hmm_contract_and_simplex(sample_feature_data):
    model = FrequentistHMM(n_regimes=5, n_iter=30, random_seed=42)
    model.fit(sample_feature_data)
    
    probs = model.predict_proba(sample_feature_data)
    
    # 1. Output shape must be (N, 5)
    assert probs.shape == (len(sample_feature_data), 5)
    
    # 2. Probability axioms: non-negative and row sums strictly 1.0
    assert (probs >= 0.0).all()
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    
    # 3. Diagnostics
    diag = model.diagnostics()
    assert "log_likelihood" in diag
    assert "aic" in diag
    assert "bic" in diag
    assert len(diag["transition_matrix"]) == 5


def test_bayesian_hmm_mcmc_and_uncertainty(sample_feature_data):
    # Fast test configuration: 40 iterations, 10 burn-in, 2 chains
    model = BayesianHMM(
        n_regimes=5,
        n_iter=40,
        burn_in=10,
        n_chains=2,
        sticky_kappa=5.0,
        random_seed=42,
    )
    model.fit(sample_feature_data)
    
    mean_probs, epistemic, aleatoric = model.predict_proba_posterior(sample_feature_data)
    
    # Shape and simplex
    assert mean_probs.shape == (len(sample_feature_data), 5)
    assert np.allclose(np.sum(mean_probs, axis=1), 1.0, atol=1e-5)
    
    # Uncertainty invariants
    assert len(epistemic) == len(sample_feature_data)
    assert len(aleatoric) == len(sample_feature_data)
    assert (epistemic >= -1e-6).all()
    assert (aleatoric >= -1e-6).all()
    
    # Diagnostics check R-hat and ESS
    diag = model.diagnostics()
    assert "gelman_rubin_r_hat" in diag
    assert "effective_sample_size_ess" in diag


def test_rs_var_hamilton_filter(sample_feature_data):
    model = RegimeSwitchingVAR(n_regimes=5, lags=1, n_iter=15, random_seed=42)
    model.fit(sample_feature_data)
    
    probs = model.predict_proba(sample_feature_data)
    assert probs.shape == (len(sample_feature_data), 5)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    
    diag = model.diagnostics()
    assert "ar_lags" in diag
    assert diag["ar_lags"] == 1


def test_bayesian_deep_learning_mc_dropout(sample_feature_data):
    model = BayesianDeepLearningModel(
        n_regimes=5,
        n_mc_samples=25,
        n_ensemble=3,
        epochs=30,
        random_seed=42,
    )
    model.fit(sample_feature_data)
    
    probs, epistemic, aleatoric = model.predict_proba_with_uncertainty(sample_feature_data)
    assert probs.shape == (len(sample_feature_data), 5)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    
    # Epistemic and aleatoric non-negativity
    assert (epistemic >= -1e-6).all()
    assert (aleatoric >= -1e-6).all()


def test_chronos_foundation_probing_adapter(sample_feature_data):
    adapter = ChronosRegimeAdapter(context_length=32, embedding_dim=8, random_seed=42)
    adapter.fit(sample_feature_data)
    
    probs = adapter.predict_proba(sample_feature_data)
    assert probs.shape == (len(sample_feature_data), 5)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    
    diag = adapter.diagnostics()
    assert "probing_classifier_accuracy" in diag
    assert "regime_separation" in diag
    assert "silhouette_score" in diag["regime_separation"]


def test_regime_duration_analyzer():
    analyzer = RegimeDurationAnalyzer()
    
    # Synthetic regime sequence with persistence: 0, 0, 0, 1, 1, 4, 4, 4, 4
    seq = np.array([0, 0, 0, 1, 1, 4, 4, 4, 4, 2, 2, 3])
    spells = analyzer.extract_durations(seq)
    
    assert spells[0] == [3]
    assert spells[1] == [2]
    assert spells[4] == [4]
    
    # Audit sequence table
    table = analyzer.analyze_sequence(seq)
    assert len(table) == 5
    assert "regime" in table.columns
    assert "reject_geometric_null" in table.columns
