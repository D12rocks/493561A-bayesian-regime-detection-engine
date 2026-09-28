"""
Forensic Audit of FastAPI Production Service Endpoints.

Calls:
1. GET /regime/health
2. POST /regime/score
3. GET /regime/explanation
4. GET /regime/audit/{date}
"""

import json
from fastapi.testclient import TestClient
from src.service.api import app


def test_api_service_endpoints() -> None:
    client = TestClient(app)

    print("--- 1. Testing GET /regime/health ---")
    resp_health = client.get("/regime/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    print("Health response:", json.dumps(health_data, indent=2))
    assert health_data["status"] == "HEALTHY"
    assert "two_speed_reconciliation" in health_data

    print("\n--- 2. Testing POST /regime/score ---")
    resp_score = client.post("/regime/score", json={"timestamp": "2024-12-30"})
    assert resp_score.status_code == 200
    score_data = resp_score.json()
    print("Score response:", json.dumps(score_data, indent=2))
    assert "predicted_regime" in score_data
    assert "regime_probabilities" in score_data
    assert "conformal_prediction_set" in score_data
    assert "changepoint_probability" in score_data

    print("\n--- 3. Testing GET /regime/explanation ---")
    resp_exp = client.get("/regime/explanation?as_of=2024-12-30")
    assert resp_exp.status_code == 200
    exp_data = resp_exp.json()
    print("Explanation response:", json.dumps(exp_data, indent=2))
    assert "natural_language_rationale" in exp_data
    assert "top_feature_drivers" in exp_data

    print("\n--- 4. Testing GET /regime/audit/2020-03-23 ---")
    resp_audit = client.get("/regime/audit/2020-03-23")
    assert resp_audit.status_code == 200
    audit_data = resp_audit.json()
    print("Audit response:", json.dumps(audit_data, indent=2))
    assert audit_data["requested_timestamp"] == "2020-03-23"
    assert "individual_model_opinions" in audit_data

    print("\nALL API SERVICE ENDPOINTS EMPIRICALLY VERIFIED AND OPERATIONAL!")


if __name__ == "__main__":
    test_api_service_endpoints()
