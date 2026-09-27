"""
Integration Test for Core Scaffolding & End-to-End Prediction Flow.

Verifies that RegimePrediction, UncertaintyMetrics, Conformal Sets,
and Investment Committee reporting integrate cleanly without contract violations.
"""

from datetime import datetime
import pytest

from src.models.contracts import (
    ModelLineage,
    RegimeLabel,
    RegimePrediction,
    RegimeProbabilities,
    UncertaintyMetrics,
)
from src.reporting.committee import InvestmentCommitteePackGenerator
from src.simulation.base import TailRiskMetrics


@pytest.mark.integration
def test_end_to_end_scaffolding_contract_flow() -> None:
    # 1. Simulate output of calibrated ensemble
    probs = RegimeProbabilities(
        risk_on=0.70,
        late_cycle=0.15,
        transitional=0.05,
        post_shock=0.05,
        risk_off=0.05,
    )
    assert probs.validate()

    uncertainty = UncertaintyMetrics(
        predictive_uncertainty=0.88,
        epistemic_uncertainty=0.12,
        aleatoric_uncertainty=0.76,
    )
    assert uncertainty.validate()

    lineage = ModelLineage(
        model_name="Ensemble_BMA_Stacking",
        model_version="1.0.0",
        feature_snapshot_id="feat_snap_v1_001",
        data_snapshot_id="data_snap_v1_001",
        training_period="2009-2017",
        random_seed=42,
    )

    pred = RegimePrediction(
        timestamp=datetime(2023, 10, 1, 15, 30),
        regime_probabilities=probs,
        predicted_regime=RegimeLabel.RISK_ON,
        uncertainty=uncertainty,
        conformal_prediction_set=[RegimeLabel.RISK_ON],
        changepoint_probability=0.02,
        lineage=lineage,
    )

    # 2. Simulate forward tail-risk metrics
    tail_risk = TailRiskMetrics(
        horizon_days=21,
        var_95=-0.042,
        cvar_95=-0.061,
        var_99=-0.078,
        cvar_99=-0.098,
        expected_return=0.015,
        path_volatility=0.13,
    )

    # 3. Generate Investment Committee Briefing
    generator = InvestmentCommitteePackGenerator()
    briefing = generator.generate_briefing(
        prediction=pred,
        tail_risk=tail_risk,
        allocation_weights={"equity_beta": 1.0, "cash_hedge": 0.0},
    )

    assert briefing.current_regime == "Risk-On"
    assert briefing.regime_confidence == 0.70
    assert briefing.forward_var_95 == -0.042
    assert "Constructive Risk-On Stance" in briefing.commentary

    json_output = briefing.to_json()
    assert '"current_regime": "Risk-On"' in json_output
