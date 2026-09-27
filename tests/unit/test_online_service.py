"""
Unit Tests for Online Inference, Particle Filtering, BOCPD, and FastAPI Service.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.online.particle_filter import BootstrapParticleFilter
from src.online.bocpd import BayesianOnlineChangepointDetector
from src.online.base import TwoSpeedReconciler
from src.models.contracts import RegimeProbabilities
from src.service.api import app


def test_bootstrap_particle_filter():
    pf = BootstrapParticleFilter(n_particles=500, random_seed=42)
    obs = np.array([0.015, 0.14])  # Positive return, low vol
    
    probs = pf.update(obs)
    assert probs.validate()
    p_arr = probs.to_array()
    assert len(p_arr) == 5
    assert np.isclose(np.sum(p_arr), 1.0, atol=1e-4)


def test_bocpd_changepoint_detector():
    detector = BayesianOnlineChangepointDetector(hazard_rate=50.0)
    
    # Send quiet series
    for _ in range(30):
        cp, _ = detector.update(0.001)
        assert 0.0 <= cp <= 1.0

    # Inject extreme shock
    cp_shock, _ = detector.update(-0.10)
    assert 0.0 <= cp_shock <= 1.0


def test_two_speed_reconciliation():
    reconciler = TwoSpeedReconciler(max_allowable_kl=0.25)
    
    p_on = RegimeProbabilities(0.50, 0.20, 0.15, 0.10, 0.05)
    p_batch_close = RegimeProbabilities(0.48, 0.22, 0.15, 0.10, 0.05)
    res_close = reconciler.reconcile(p_on, p_batch_close)
    assert res_close["is_consistent"] == True
    assert res_close["requires_ad_hoc_batch_trigger"] == False

    p_batch_divergent = RegimeProbabilities(0.05, 0.05, 0.10, 0.20, 0.60)
    res_div = reconciler.reconcile(p_on, p_batch_divergent)
    assert res_div["is_consistent"] == False
    assert res_div["requires_ad_hoc_batch_trigger"] == True


def test_fastapi_endpoints():
    client = TestClient(app)

    # 1. Health
    resp = client.get("/regime/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert "two_speed_reconciliation" in data

    # 2. Score
    score_resp = client.post("/regime/score", json={"observation": [0.005, 0.15]})
    assert score_resp.status_code == 200
    s_data = score_resp.json()
    assert "predicted_regime" in s_data
    assert "regime_probabilities" in s_data
    assert "conformal_prediction_set" in s_data

    # 3. Explanation
    exp_resp = client.get("/regime/explanation")
    assert exp_resp.status_code == 200
    e_data = exp_resp.json()
    assert "natural_language_rationale" in e_data
    assert "top_feature_drivers" in e_data

    # 4. Audit
    audit_resp = client.get("/regime/audit/2020-03-23")
    assert audit_resp.status_code == 200
    a_data = audit_resp.json()
    assert "data_snapshot_sha256" in a_data
    assert "individual_model_opinions" in a_data
