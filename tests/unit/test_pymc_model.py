"""
Unit test for PyMC Bayesian Regime Model with Dirichlet priors and NUTS sampling.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.pymc_hmm import PyMCBayesianRegimeModel


def test_pymc_bayesian_regime_model_nuts_fit():
    rng = np.random.RandomState(42)
    n_obs = 60
    returns = rng.randn(n_obs) * 0.02
    vol = np.abs(returns)
    df = pd.DataFrame({"nifty_ret_1d": returns, "vix": vol})

    model = PyMCBayesianRegimeModel(
        n_regimes=5,
        n_draws=50,
        n_tune=50,
        n_chains=2,
        random_seed=42,
    )
    model.fit(df)
    assert model.is_fitted
    assert model.trace is not None

    diag = model.diagnostics()
    assert diag["sampler"] == "NUTS (No-U-Turn Sampler)"
    assert diag["inference_engine"] == "PyMC"
    assert "r_hat_max" in diag
    assert "divergences" in diag

    probs = model.predict_proba(df)
    assert probs.shape == (n_obs, 5)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
