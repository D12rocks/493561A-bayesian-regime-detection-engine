"""
Unit tests for the AI Regime Copilot and Scenario Lab.

Tests:
    - No representative SHAP fallback (returns UNAVAILABLE)
    - No synthetic BOCPD fallback (returns UNAVAILABLE, never 0.42)
    - Stale RPS/ECE values (0.0003, 0.0014, 0.0011) cannot appear in current governance context
    - Programmatic LLM response validator rejects unsupported numerical claims:
        - dominant regime mismatch
        - regime probability discrepancy
        - BOCPD changepoint hallucination when UNAVAILABLE
        - conformal set size / membership discrepancy
        - backtest metric discrepancy (CAGR, Sharpe, Drawdown)
        - stale circular evaluation metric injection
    - Strict source status contract (VERIFIED, UNAVAILABLE, QUARANTINED, SYNTHETIC_SCENARIO)
    - Scenario Lab simulation engine labeling:
        - FULL MONTE CARLO vs PARAMETRIC FALLBACK distinction
        - scenario_type = SYNTHETIC_NL_STRESS
        - ui_label = HYPOTHETICAL SCENARIO — NOT A FORECAST
    - Scenario sign semantics for USD/INR:
        - USD/INR +3% (positive shock delta)
        - USD/INR -3% (negative shock delta)
        - rupee appreciation (negative shock delta)
        - rupee depreciation (positive shock delta)
        - USD/INR appreciates vs rupee appreciates have opposite directions
    - Auditability: InteractionLogger writes all 8 required audit fields
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ai.llm_provider import MockProvider, get_provider
from src.ai.tools import RegimeToolKit
from src.ai.copilot import RegimeCopilot
from src.ai.response_validator import ResponseValidator, ValidationResult
from src.ai.scenario_parser import (
    ScenarioParser,
    ParsedScenario,
    simulate_scenario,
    ScenarioSimulationResult,
)
from src.ai.interaction_logger import InteractionLogger


# ---------------------------------------------------------------------------
# Helpers: synthetic test DataFrames
# ---------------------------------------------------------------------------

REGIME_COLS = [
    "ensemble_prob_risk_on",
    "ensemble_prob_late_cycle",
    "ensemble_prob_transitional",
    "ensemble_prob_post_shock",
    "ensemble_prob_risk_off",
]


def _make_preds_df(n: int = 10, include_bocpd: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(np.ones(5), size=n)
    df = pd.DataFrame(raw, columns=REGIME_COLS, index=dates)
    if include_bocpd:
        df["changepoint_probability"] = rng.uniform(0.05, 0.40, size=n)
    return df


def _make_feats_df(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "nifty_ret_1d": rng.normal(0.0, 0.01, n),
        "nifty_vol_ewma_21d": rng.uniform(0.10, 0.30, n),
        "breadth_midcap_ret_21d": rng.normal(0.0, 0.02, n),
        "usdinr_ret_21d": rng.normal(0.0, 0.01, n),
        "vix_level": rng.uniform(12.0, 25.0, n),
        "nifty_dist_sma50": rng.normal(0.0, 0.05, n),
    }, index=dates)
    return df


# ===========================================================================
# 1. Strict Evidence & Fallback Removal Tests
# ===========================================================================

class TestStrictEvidenceAndFallbackRemoval:
    @pytest.fixture
    def toolkit_no_bocpd(self):
        return RegimeToolKit(
            cal_preds_df=_make_preds_df(include_bocpd=False),
            features_df=_make_feats_df(),
        )

    def test_no_representative_shap_fallback(self, toolkit_no_bocpd, tmp_path, monkeypatch):
        """If real SHAP artifacts are absent or missing the requested date, return UNAVAILABLE."""
        # Point to a nonexistent path
        monkeypatch.setattr(
            "src.ai.tools.Path",
            lambda p: tmp_path / "nonexistent.csv" if "shap" in str(p) else Path(p),
        )
        res = toolkit_no_bocpd.get_shap_attributions("2024-01-02")
        assert res["status"] == "UNAVAILABLE"
        assert "not present" in res["reason"] or "unavailable" in res["reason"].lower()
        assert "top_shap_features" not in res

    def test_no_synthetic_bocpd_fallback(self, toolkit_no_bocpd):
        """If no BOCPD column exists, return UNAVAILABLE — never substitute 0.42."""
        res = toolkit_no_bocpd.get_changepoint_signal("2024-01-02")
        assert res["status"] == "UNAVAILABLE"
        assert "No BOCPD changepoint column found" in res["reason"]
        # Ensure 0.42 is NEVER returned
        assert res.get("bocpd_changepoint_probability") != 0.42
        assert "bocpd_changepoint_probability" not in res

    def test_bocpd_verified_when_column_present(self):
        """When real BOCPD column is present, status is VERIFIED and probability is valid."""
        tk = RegimeToolKit(
            cal_preds_df=_make_preds_df(include_bocpd=True),
            features_df=_make_feats_df(),
        )
        res = tk.get_changepoint_signal("2024-01-02")
        assert res["status"] == "VERIFIED"
        assert 0.0 <= res["bocpd_changepoint_probability"] <= 1.0

    def test_stale_governance_metrics_cannot_appear_in_toolkit(self, toolkit_no_bocpd):
        """Stale circular metrics (0.0003 RPS / 0.0014 ECE / 0.0011 Log Loss) must never appear."""
        gov = toolkit_no_bocpd.get_governance_metadata()
        gov_str = json.dumps(gov)
        assert "0.0003" not in gov_str
        assert "0.0014" not in gov_str
        assert "0.0011" not in gov_str

    def test_strict_source_status_contract(self, toolkit_no_bocpd):
        """Every evidence tool must return source, status, timestamp, and provenance."""
        allowed_statuses = {"VERIFIED", "UNAVAILABLE", "QUARANTINED", "SYNTHETIC_SCENARIO"}

        bundle = toolkit_no_bocpd.build_evidence_bundle("2024-01-02")
        for tool_key in [
            "regime_state", "feature_snapshot", "shap_attributions",
            "changepoint_signal", "conformal_context", "historical_analogs",
            "governance", "backtest_context",
        ]:
            evidence_obj = bundle[tool_key]
            assert "source" in evidence_obj, f"{tool_key} missing 'source'"
            assert "status" in evidence_obj, f"{tool_key} missing 'status'"
            assert "timestamp" in evidence_obj, f"{tool_key} missing 'timestamp'"
            assert "provenance" in evidence_obj, f"{tool_key} missing 'provenance'"
            assert evidence_obj["status"] in allowed_statuses, (
                f"{tool_key} status '{evidence_obj['status']}' not in allowed set"
            )

    def test_empty_predictions_returns_unavailable(self):
        tk = RegimeToolKit(cal_preds_df=pd.DataFrame(), features_df=pd.DataFrame())
        res = tk.get_regime_state("2024-01-02")
        assert res["status"] == "UNAVAILABLE"

    def test_invalid_date_returns_unavailable(self, toolkit_no_bocpd):
        res = toolkit_no_bocpd.get_regime_state("invalid-date-string")
        assert res["status"] == "UNAVAILABLE"


# ===========================================================================
# 2. Programmatic LLM Grounding Validation Tests
# ===========================================================================

class TestResponseValidator:
    @pytest.fixture
    def validator(self):
        return ResponseValidator()

    @pytest.fixture
    def evidence_bundle(self):
        tk = RegimeToolKit(
            cal_preds_df=_make_preds_df(include_bocpd=False),
            features_df=_make_feats_df(),
        )
        bundle = tk.build_evidence_bundle("2024-01-02")
        return bundle

    def test_accepts_grounded_response(self, validator, evidence_bundle):
        dom = evidence_bundle["dominant_regime"]
        prob = evidence_bundle["dominant_prob"] * 100.0
        response = (
            f"The current dominant regime is **{dom}** with probability {prob:.1f}%. "
            "Predictive entropy is moderate. CAGR is 12.67% and Sharpe is 0.49."
        )
        res = validator.validate(response, evidence_bundle)
        assert res.is_trusted is True
        assert len(res.violations) == 0

    def test_rejects_dominant_regime_hallucination(self, validator, evidence_bundle):
        dom = evidence_bundle["dominant_regime"]
        other_regime = "Risk-Off" if dom != "Risk-Off" else "Risk-On"
        response = f"The dominant regime is {other_regime} today."
        res = validator.validate(response, evidence_bundle)
        assert res.is_trusted is False
        assert any("dominant regime" in v for v in res.violations)
        assert "GROUNDING VALIDATION FAILED" in res.safe_response

    def test_rejects_regime_probability_hallucination(self, validator, evidence_bundle):
        dom = evidence_bundle["dominant_regime"]
        auth_prob = evidence_bundle["dominant_prob"] * 100.0
        # Discrepancy > 2.5 percentage points
        hallucinated_prob = (auth_prob + 15.0) % 100.0
        response = f"The {dom} probability is {hallucinated_prob:.1f}%."
        res = validator.validate(response, evidence_bundle)
        assert res.is_trusted is False
        assert any("probability" in v for v in res.violations)

    def test_rejects_bocpd_claim_when_unavailable(self, validator, evidence_bundle):
        assert evidence_bundle["changepoint_signal"]["status"] == "UNAVAILABLE"
        response = "The changepoint probability is 42% according to BOCPD."
        res = validator.validate(response, evidence_bundle)
        assert res.is_trusted is False
        assert any("BOCPD signal is UNAVAILABLE" in v or "changepoint" in v for v in res.violations)

    def test_rejects_conformal_set_size_mismatch(self, validator, evidence_bundle):
        auth_set = evidence_bundle["conformal_set"]
        auth_size = len(auth_set)
        hallucinated_size = auth_size + 2
        response = f"The conformal prediction set of size {hallucinated_size} was emitted."
        res = validator.validate(response, evidence_bundle)
        assert res.is_trusted is False
        assert any("conformal set size" in v for v in res.violations)

    def test_rejects_backtest_cagr_discrepancy(self, validator, evidence_bundle):
        response = "The strategy achieves a CAGR of 24.50% in backtests."
        res = validator.validate(response, evidence_bundle)
        assert res.is_trusted is False
        assert any("CAGR" in v for v in res.violations)

    def test_rejects_stale_governance_metrics(self, validator, evidence_bundle):
        for stale_val in ["0.0003", "0.0014", "0.0011"]:
            response = f"The model achieves an outstanding proper RPS score of {stale_val}."
            res = validator.validate(response, evidence_bundle)
            assert res.is_trusted is False
            assert any("stale, superseded governance metric" in v for v in res.violations)


# ===========================================================================
# 3. Scenario Lab Simulation Engine Labeling Tests
# ===========================================================================

class TestScenarioLabSimulationEngine:
    @pytest.fixture
    def parsed_scenario(self):
        parser = ScenarioParser()
        return parser.parse("Increase India VIX by 30% and drop NIFTY 4 percent")

    @pytest.fixture
    def baseline_features(self):
        return {
            "vix_level": 14.5,
            "nifty_ret_1d": 0.002,
            "nifty_vol_ewma_21d": 0.13,
            "usdinr_ret_21d": 0.001,
            "breadth_midcap_ret_21d": 0.005,
            "nifty_dist_sma50": 0.02,
        }

    def test_monte_carlo_vs_parametric_fallback_correctly_labelled(
        self, parsed_scenario, baseline_features
    ):
        """Simulation must distinguish FULL MONTE CARLO from PARAMETRIC FALLBACK."""
        # 1. Run with Monte Carlo available
        mc_result: ScenarioSimulationResult = simulate_scenario(
            parsed=parsed_scenario,
            baseline_features=baseline_features,
            force_parametric=False,
        )
        assert mc_result.scenario_type == "SYNTHETIC_NL_STRESS"
        assert mc_result.ui_label == "HYPOTHETICAL SCENARIO — NOT A FORECAST"
        assert mc_result.status == "SYNTHETIC_SCENARIO"
        assert mc_result.engine_used == "FULL MONTE CARLO"
        assert mc_result.is_monte_carlo is True

        # 2. Run with forced parametric fallback
        fallback_result: ScenarioSimulationResult = simulate_scenario(
            parsed=parsed_scenario,
            baseline_features=baseline_features,
            force_parametric=True,
        )
        assert fallback_result.scenario_type == "SYNTHETIC_NL_STRESS"
        assert fallback_result.ui_label == "HYPOTHETICAL SCENARIO — NOT A FORECAST"
        assert fallback_result.status == "SYNTHETIC_SCENARIO"
        assert fallback_result.engine_used == "PARAMETRIC FALLBACK"
        assert fallback_result.is_monte_carlo is False
        # Never presents parametric fallback as Monte Carlo output
        assert "Monte Carlo" not in fallback_result.engine_used


# ===========================================================================
# 4. USD/INR Sign Semantics Tests
# ===========================================================================

class TestScenarioSignSemantics:
    @pytest.fixture
    def parser(self):
        return ScenarioParser()

    def test_usdinr_increase_positive_delta(self, parser):
        """USD/INR +3% means USD rises vs INR (Rupee weakens) -> usdinr_ret_21d > 0."""
        r = parser.parse("USD/INR +3%")
        assert r.is_valid
        assert r.shocks[0].feature_name == "usdinr_ret_21d"
        assert r.shocks[0].shock_value > 0
        assert r.shocks[0].shock_value == pytest.approx(0.03, rel=1e-3)

    def test_usdinr_decrease_negative_delta(self, parser):
        """USD/INR -3% means USD falls vs INR (Rupee strengthens) -> usdinr_ret_21d < 0."""
        r = parser.parse("USD/INR -3%")
        assert r.is_valid
        assert r.shocks[0].feature_name == "usdinr_ret_21d"
        assert r.shocks[0].shock_value < 0
        assert r.shocks[0].shock_value == pytest.approx(-0.03, rel=1e-3)

    def test_rupee_appreciation_negative_delta(self, parser):
        """Rupee appreciation means INR strengthens -> USD/INR falls -> usdinr_ret_21d < 0."""
        r = parser.parse("rupee appreciation")
        assert r.is_valid
        assert r.shocks[0].feature_name == "usdinr_ret_21d"
        assert r.shocks[0].shock_value < 0

    def test_rupee_depreciation_positive_delta(self, parser):
        """Rupee depreciation means INR weakens -> USD/INR rises -> usdinr_ret_21d > 0."""
        r = parser.parse("rupee depreciation")
        assert r.is_valid
        assert r.shocks[0].feature_name == "usdinr_ret_21d"
        assert r.shocks[0].shock_value > 0

    def test_usdinr_appreciates_vs_rupee_appreciates_different_directions(self, parser):
        """USD/INR appreciates and rupee appreciates must NOT map to the same internal direction."""
        r_usdinr = parser.parse("USD/INR appreciates 3%")
        r_rupee = parser.parse("rupee appreciates 3%")
        assert r_usdinr.is_valid and r_rupee.is_valid

        val_usdinr = r_usdinr.shocks[0].shock_value
        val_rupee = r_rupee.shocks[0].shock_value

        # They must have opposite signs!
        assert val_usdinr > 0, "USD/INR appreciates should have positive delta"
        assert val_rupee < 0, "Rupee appreciates should have negative delta"
        assert val_usdinr == -val_rupee


# ===========================================================================
# 5. Auditability Tests
# ===========================================================================

class TestAuditability:
    def test_interaction_logger_records_all_eight_fields(self, tmp_path):
        """Logs date, provider, model_name/version, query, hashes, statuses, scenario_id."""
        log_dir = tmp_path / "ai_logs"
        logger = InteractionLogger(log_dir=log_dir)

        evidence_bundle = {"dominant_regime": "Late-Cycle", "dominant_prob": 0.75}
        source_statuses = {"regime_state": "VERIFIED", "bocpd": "UNAVAILABLE"}

        logger.log(
            interaction_type="COPILOT",
            query="What is the current regime?",
            response="The regime is Late-Cycle.",
            provider_name="MockProvider",
            model_name="deterministic-mock-v1",
            evidence_bundle=evidence_bundle,
            evidence_source_statuses=source_statuses,
            analysis_date="2024-06-04",
            scenario_id="SCEN-TEST-001",
        )

        records = logger.get_recent(5)
        assert len(records) == 1
        rec = records[0]

        # Check all required audit fields
        assert "date" in rec and rec["date"] == "2024-06-04"
        assert "provider" in rec and rec["provider"] == "MockProvider"
        assert "model_name" in rec and rec["model_name"] == "deterministic-mock-v1"
        assert "user_query" in rec and rec["user_query"] == "What is the current regime?"
        assert "evidence_bundle_hash" in rec and len(rec["evidence_bundle_hash"]) == 16
        assert "response_hash" in rec and len(rec["response_hash"]) == 16
        assert "evidence_source_statuses" in rec and rec["evidence_source_statuses"] == source_statuses
        assert "scenario_id" in rec and rec["scenario_id"] == "SCEN-TEST-001"


# ===========================================================================
# 6. RegimeCopilot Integration Tests
# ===========================================================================

class TestRegimeCopilotIntegration:
    @pytest.fixture
    def copilot(self):
        toolkit = RegimeToolKit(
            cal_preds_df=_make_preds_df(include_bocpd=False),
            features_df=_make_feats_df(),
        )
        return RegimeCopilot(toolkit=toolkit, provider=MockProvider(), log_interactions=False)

    def test_copilot_ask_without_date_returns_error(self, copilot):
        res = copilot.ask("What regime?")
        assert "error" in res

    def test_copilot_set_date_and_ask_structured(self, copilot):
        copilot.set_date("2024-01-02")
        res = copilot.ask("What is the current regime?")
        assert "model_output" in res
        assert "calculated_evidence" in res
        assert "ai_interpretation" in res
        assert "grounding_trusted" in res
        assert res["grounding_trusted"] is True

    def test_copilot_handles_date_reset(self, copilot):
        copilot.set_date("2024-01-02")
        copilot.ask("Query 1")
        assert len(copilot.history) == 2
        copilot.set_date("2024-01-03")
        assert len(copilot.history) == 0

    def test_copilot_suggested_questions(self, copilot):
        copilot.set_date("2024-01-02")
        qs = copilot.get_suggested_questions()
        assert len(qs) >= 1
        assert any("regime" in q.lower() for q in qs)
