"""
Unit tests for core data and model contracts.
"""

from datetime import datetime
import numpy as np
import pytest

from src.models.contracts import (
    ModelLineage,
    RegimeLabel,
    RegimePrediction,
    RegimeProbabilities,
    UncertaintyMetrics,
)


@pytest.mark.unit
def test_regime_probabilities_valid() -> None:
    probs = RegimeProbabilities(
        risk_on=0.5,
        late_cycle=0.2,
        transitional=0.1,
        post_shock=0.1,
        risk_off=0.1,
    )
    assert probs.validate() is True
    arr = probs.to_array()
    assert len(arr) == 5
    assert np.isclose(np.sum(arr), 1.0)


@pytest.mark.unit
def test_regime_probabilities_invalid_sum() -> None:
    probs = RegimeProbabilities(
        risk_on=0.5,
        late_cycle=0.2,
        transitional=0.1,
        post_shock=0.1,
        risk_off=0.5,  # Sum = 1.4
    )
    assert probs.validate() is False


@pytest.mark.unit
def test_regime_probabilities_from_array() -> None:
    arr = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs = RegimeProbabilities.from_array(arr)
    assert probs.validate() is True
    assert np.isclose(probs.risk_on, 0.2)


@pytest.mark.unit
def test_uncertainty_metrics_validation() -> None:
    valid_u = UncertaintyMetrics(
        predictive_uncertainty=1.2,
        epistemic_uncertainty=0.3,
        aleatoric_uncertainty=0.9,
    )
    assert valid_u.validate() is True

    invalid_u = UncertaintyMetrics(
        predictive_uncertainty=-0.5,
        epistemic_uncertainty=0.3,
        aleatoric_uncertainty=0.9,
    )
    assert invalid_u.validate() is False


@pytest.mark.unit
def test_regime_prediction_initialization(valid_regime_prediction: RegimePrediction) -> None:
    assert valid_regime_prediction.predicted_regime == RegimeLabel.RISK_ON
    assert valid_regime_prediction.changepoint_probability == 0.04
    assert len(valid_regime_prediction.conformal_prediction_set) == 2


@pytest.mark.unit
def test_regime_prediction_invalid_changepoint(valid_regime_probabilities: RegimeProbabilities) -> None:
    lineage = ModelLineage(
        model_name="M",
        model_version="1",
        feature_snapshot_id="F",
        data_snapshot_id="D",
        training_period="P",
        random_seed=42,
    )
    uncertainty = UncertaintyMetrics(1.0, 0.2, 0.8)
    with pytest.raises(ValueError, match="Changepoint probability"):
        RegimePrediction(
            timestamp=datetime.now(),
            regime_probabilities=valid_regime_probabilities,
            predicted_regime=RegimeLabel.RISK_ON,
            uncertainty=uncertainty,
            conformal_prediction_set=[RegimeLabel.RISK_ON],
            changepoint_probability=1.5,  # Invalid: > 1.0
            lineage=lineage,
        )
