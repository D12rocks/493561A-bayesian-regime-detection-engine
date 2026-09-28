"""
Unit tests for disaggregated Bayesian Deep Learning models:
1. MC Dropout Network
2. Variational BNN (Bayes by Backprop)
3. Deep Ensemble
"""

import numpy as np
import pandas as pd
import pytest

from src.models.bayesian_dl import BayesianDeepLearningModel
from src.models.variational_bnn import VariationalBNNModel
from src.models.deep_ensemble import DeepEnsembleModel


def test_mc_dropout_model():
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    model = BayesianDeepLearningModel(n_regimes=5, n_mc_samples=20, n_ensemble=2, epochs=10, random_seed=42)
    model.fit(df)
    assert model.is_fitted
    probs, epi, alea = model.predict_proba_with_uncertainty(df)
    assert probs.shape == (50, 5)
    assert len(epi) == 50
    assert len(alea) == 50


def test_variational_bnn_bayes_by_backprop():
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    model = VariationalBNNModel(n_regimes=5, hidden_dim=16, n_mc_samples=20, epochs=10, random_seed=42)
    model.fit(df)
    assert model.is_fitted
    probs, epi, alea = model.predict_proba_with_uncertainty(df)
    assert probs.shape == (50, 5)
    assert len(epi) == 50
    assert len(alea) == 50
    diag = model.diagnostics()
    assert "Variational BNN" in diag["model_type"]


def test_deep_ensemble():
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    model = DeepEnsembleModel(n_regimes=5, n_members=3, hidden_dim=16, epochs=10, random_seed=42)
    model.fit(df)
    assert model.is_fitted
    probs, epi, alea = model.predict_proba_with_uncertainty(df)
    assert probs.shape == (50, 5)
    assert len(epi) == 50
    assert len(alea) == 50
    diag = model.diagnostics()
    assert diag["n_members"] == 3
