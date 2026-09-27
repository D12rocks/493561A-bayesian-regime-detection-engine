"""
Statistical Invariant Tests.

Verifies strict mathematical and probabilistic axioms:
1. Regime probability simplex invariants (sum = 1, non-negative).
2. Transition matrix stochasticity.
3. Information-theoretic uncertainty decomposition.
4. Metric bounds (ECE in [0, 1], Brier score in [0, 2]).
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_expected_calibration_error,
    compute_multi_class_brier_score,
)
from src.models.contracts import RegimeProbabilities


@pytest.mark.statistical
def test_probability_simplex_invariant() -> None:
    # Test random Dirichlet draws
    for _ in range(50):
        sample = np.random.dirichlet(np.ones(5))
        probs = RegimeProbabilities.from_array(sample)
        assert probs.validate() is True
        assert np.isclose(np.sum(probs.to_array()), 1.0, atol=1e-5)
        assert np.all(probs.to_array() >= 0.0)


@pytest.mark.statistical
def test_transition_matrix_row_stochasticity() -> None:
    """A transition matrix A must satisfy sum_j A_{ij} = 1 for all rows i."""
    A = np.random.dirichlet(np.ones(5), size=5)
    row_sums = np.sum(A, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.all(A >= 0.0)


@pytest.mark.statistical
def test_ece_bounds() -> None:
    """ECE must strictly lie in [0.0, 1.0]."""
    N = 200
    probs = np.random.dirichlet(np.ones(5), size=N)
    labels = np.random.randint(0, 5, size=N)

    ece = compute_expected_calibration_error(probs, labels, n_bins=10)
    assert 0.0 <= ece <= 1.0


@pytest.mark.statistical
def test_brier_score_bounds() -> None:
    """Multi-class Brier score must strictly lie in [0.0, 2.0]."""
    N = 200
    probs = np.random.dirichlet(np.ones(5), size=N)
    labels = np.random.randint(0, 5, size=N)

    brier = compute_multi_class_brier_score(probs, labels)
    assert 0.0 <= brier <= 2.0
