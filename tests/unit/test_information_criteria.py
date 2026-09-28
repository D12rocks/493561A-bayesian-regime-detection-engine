"""
Unit tests for Bayesian Information Criteria (WAIC and PSIS-LOO).
"""

import numpy as np
import pytest

from src.evaluation.information_criteria import BayesianInformationCriteria


def test_waic_and_psis_loo_computation():
    rng = np.random.RandomState(42)
    N = 100
    D = 2
    S = 50
    K = 5

    X = rng.randn(N, D)
    post_means = rng.randn(S, K, D) * 0.1
    post_covs = np.array([[[np.eye(D) for _ in range(K)] for _ in range(S)]]).reshape(S, K, D, D)
    post_states = rng.randint(0, K, size=(S, N))

    log_lik = BayesianInformationCriteria.compute_pointwise_log_likelihood(
        X, post_means, post_covs, post_states
    )
    assert log_lik.shape == (1, S, N)

    # Compute WAIC
    waic_res = BayesianInformationCriteria.compute_waic_from_log_lik(log_lik)
    assert "waic" in waic_res
    assert "p_waic" in waic_res
    assert waic_res["p_waic"] >= 0.0

    # Compute PSIS-LOO
    loo_res = BayesianInformationCriteria.compute_arviz_loo(log_lik)
    assert "elpd_loo" in loo_res
    assert "pareto_k_good_pct" in loo_res
    assert loo_res["pareto_k_good_pct"] >= 0.0
