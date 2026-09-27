"""
Unit Tests for Explainability, Model Governance, Historical Audit Replay, and IC Briefing.
"""

import numpy as np
import pandas as pd
import pytest

from src.explainability import RegimeShapExplainer, DynamicNarrativeGenerator
from src.governance import ModelLifecycleState, ModelRegistrationCard, ModelRegistry, GovernanceMonitor
from src.audit.replay import HistoricalAuditEngine
from src.reporting.committee_brief import InvestmentCommitteeBriefGenerator


def test_shap_explainer_and_model_disagreement():
    def mock_predict(df):
        # Return probability vector based on feat0
        p = np.full((len(df), 5), 0.1)
        p[:, 0] = 0.6  # Dominant Risk-On
        return p

    explainer = RegimeShapExplainer(
        predict_fn=mock_predict,
        feature_names=["f1", "f2"],
        n_permutations=10,
    )
    instance = pd.Series({"f1": 0.05, "f2": 0.20})
    shap_vals = explainer.explain_instance(instance, target_regime_idx=0)
    assert "f1" in shap_vals
    assert "f2" in shap_vals

    # Model disagreement
    opinions = {
        "ModelA": np.array([[0.7, 0.1, 0.1, 0.05, 0.05]]),
        "ModelB": np.array([[0.65, 0.15, 0.1, 0.05, 0.05]]),
    }
    diag = RegimeShapExplainer.compute_model_disagreement(opinions)
    assert 0.0 <= diag["concordance"] <= 1.0


def test_dynamic_narrative_generation():
    features = pd.Series({"vix_level": 12.5, "nifty_dist_sma50": 0.03, "breadth_midcap_ret_21d": 0.015})
    shap = {"vix_level": 0.35, "breadth_midcap_ret_21d": 0.25, "nifty_dist_sma50": 0.15}
    opinions = {
        "Bayesian_HMM": [0.75, 0.15, 0.05, 0.03, 0.02],
        "RS_VAR": [0.70, 0.15, 0.08, 0.04, 0.03],
    }
    narrative = DynamicNarrativeGenerator.generate_explanation(
        predicted_regime="Risk-On",
        regime_probability=0.75,
        feature_values=features,
        shap_attributions=shap,
        model_opinions=opinions,
        changepoint_prob=0.04,
    )
    assert "Risk-On probability stands at 75.0%" in narrative
    assert "compressed INDIA VIX levels" in narrative
    assert "Bayesian_HMM, RS_VAR" in narrative or "confirming Risk-On dynamics" in narrative


def test_governance_psi_and_conformal_monitor():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 500)
    curr_stable = np.random.normal(0, 1, 500)
    curr_drift = np.random.normal(2.5, 1, 500)  # Significant shift

    psi_stable = GovernanceMonitor.compute_psi(ref, curr_stable)
    psi_drift = GovernanceMonitor.compute_psi(ref, curr_drift)

    assert psi_stable < 0.15
    assert psi_drift > 0.25  # Significant drift detected

    health = GovernanceMonitor.audit_conformal_health(0.92, nominal_target=0.90)
    assert health["is_healthy"] == True


def test_historical_audit_and_committee_brief():
    brief_gen = InvestmentCommitteeBriefGenerator()
    # Test replay of COVID crash: 2020-03-23
    md_brief = brief_gen.generate_brief_markdown("2020-03-23")
    assert "# INDIAN EQUITY REGIME BRIEF" in md_brief
    assert "2020-03-23" in md_brief
    assert "AUDIT_20200323" in md_brief
    assert "Conformal Prediction Set" in md_brief
    assert "Raw Market Data Snapshot SHA-256" in md_brief
