"""
Unit tests for the AI Regime Copilot and Scenario Lab.

Tests:
    - LLM provider availability detection and deterministic MockProvider
    - RegimeToolKit read-only evidence extraction and strict status contract:
        - no representative SHAP fallback (returns UNAVAILABLE)
        - no synthetic BOCPD fallback (returns UNAVAILABLE, never 0.42)
        - stale RPS/ECE values (0.0003, 0.0014, 0.0011) cannot appear in governance context
        - verified BOCPD probability when column is present
        - handling of empty predictions and invalid dates (UNAVAILABLE)
        - probability invariants, predictive entropy, and historical analogs
    - ResponseValidator programmatic LLM grounding validation:
        - accepts properly grounded responses
        - rejects dominant regime hallucination
        - rejects regime probability discrepancies (> 2.5 pp tolerance)
        - rejects BOCPD changepoint claims when UNAVAILABLE
        - rejects conformal set size / membership discrepancies
        - rejects backtest metric discrepancies (CAGR, Sharpe, Drawdown)
        - rejects stale circular evaluation metrics (0.0003, 0.0014, 0.0011)
    - ScenarioParser natural-language shock vector extraction:
        - single shocks (VIX %, NIFTY %, VIX absolute spike)
        - compound shocks ("and" / comma delimited)
        - shock direction and extreme shock capping (60 pt cap)
        - shock vector conversion and display table generation
        - preset scenario library parsing
        - USD/INR sign semantics:
            - USD/INR +3% (positive shock delta)
            - USD/INR -3% (negative shock delta)
            - rupee appreciation (negative shock delta)
            - rupee depreciation (positive shock delta)
            - USD/INR appreciates vs rupee appreciates have opposite directions
    - Scenario Lab simulation engine labeling:
        - FULL MONTE CARLO vs PARAMETRIC FALLBACK distinction
        - scenario_type = SYNTHETIC_NL_STRESS
        - ui_label = HYPOTHETICAL SCENARIO — NOT A FORECAST
    - RegimeCopilot end-to-end integration:
        - date context requirement and date-change history reset
        - structured response separation (model_output, calculated_evidence, ai_interpretation)
        - grounding trust reflection
        - suggested question generation
    - InteractionLogger auditability:
        - log writing and retrieval
        - session interaction count
        - valid JSONL record structure on disk
        - all 8 required audit fields (date, provider, model_name/version, user_query, hashes, statuses, scenario_id)
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
# 1. LLM Provider Tests
# ===========================================================================

class TestMockProvider:
    def test_always_available(self):
        p = MockProvider()
        assert p.is_available() is True

    def test_provider_name(self):
        p = MockProvider()
        assert "Deterministic" in p.provider_name or "Fallback" in p.provider_name or "Mock" in p.provider_name

    def test_chat_returns_string(self):
        p = MockProvider()
        resp = p.chat("You are an analyst.", "What is the current regime?")
        assert isinstance(resp, str)
        assert len(resp) > 10

    def test_chat_regime_question(self):
        evidence = {
            "dominant_regime": "Risk-Off",
            "dominant_prob": 0.82,
            "predictive_entropy": 0.15,
            "changepoint_probability": 0.12,
            "conformal_set": ["Risk-Off"],
            "top_shap_features": ["vix_level"],
        }
        system = f"Analyst context. ```json\n{json.dumps(evidence)}\n```"
        p = MockProvider()
        resp = p.chat(system, "What is the current regime?")
        assert "Risk-Off" in resp or "regime" in resp.lower()

    def test_chat_changepoint_question(self):
        evidence = {
            "dominant_regime": "Transitional",
            "dominant_prob": 0.55,
            "predictive_entropy": 0.60,
            "changepoint_probability": 0.45,
            "conformal_set": ["Transitional", "Risk-Off"],
            "top_shap_features": [],
        }
        system = f"```json\n{json.dumps(evidence)}\n```"
        p = MockProvider()
        resp = p.chat(system, "Is there a changepoint forming?")
        assert "changepoint" in resp.lower() or "transition" in resp.lower()

    def test_get_provider_returns_mock_when_ollama_down(self):
        with patch("src.ai.llm_provider.OllamaProvider.is_available", return_value=False):
            p = get_provider()
        assert isinstance(p, MockProvider)


# ===========================================================================
# 2. RegimeToolKit Evidence Tests (including Hardening & Status Guarantees)
# ===========================================================================

class TestRegimeToolKit:
    @pytest.fixture
    def toolkit(self):
        return RegimeToolKit(
            cal_preds_df=_make_preds_df(include_bocpd=False),
            features_df=_make_feats_df(),
        )

    def test_get_regime_state_returns_dict(self, toolkit):
        result = toolkit.get_regime_state("2024-01-02")
        assert isinstance(result, dict)
        assert "dominant_regime" in result
        assert "regime_probabilities" in result
        assert "conformal_set" in result

    def test_probabilities_sum_to_one(self, toolkit):
        result = toolkit.get_regime_state("2024-01-02")
        probs = result["regime_probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_dominant_regime_consistent(self, toolkit):
        result = toolkit.get_regime_state("2024-01-02")
        probs = result["regime_probabilities"]
        dom = result["dominant_regime"]
        assert probs[dom] == result["dominant_prob"]

    def test_conformal_set_nonempty(self, toolkit):
        result = toolkit.get_regime_state("2024-01-02")
        assert len(result["conformal_set"]) >= 1

    def test_entropy_nonnegative(self, toolkit):
        result = toolkit.get_regime_state("2024-01-02")
        assert result["predictive_entropy"] >= 0.0

    def test_get_feature_snapshot_returns_dict(self, toolkit):
        result = toolkit.get_feature_snapshot("2024-01-02")
        assert "actual_date" in result
        assert "source" in result

    def test_no_representative_shap_fallback(self, toolkit, tmp_path, monkeypatch):
        """If real SHAP artifacts are absent or missing date, return UNAVAILABLE."""
        monkeypatch.setattr(
            "src.ai.tools.Path",
            lambda p: tmp_path / "nonexistent.csv" if "shap" in str(p) else Path(p),
        )
        res = toolkit.get_shap_attributions("2024-01-02")
        assert res["status"] == "UNAVAILABLE"
        assert "top_shap_features" not in res

    def test_no_synthetic_bocpd_fallback(self, toolkit):
        """If no BOCPD column exists, return UNAVAILABLE — never substitute 0.42."""
        res = toolkit.get_changepoint_signal("2024-01-02")
        assert res["status"] == "UNAVAILABLE"
        assert res.get("bocpd_changepoint_probability") != 0.42
        assert "bocpd_changepoint_probability" not in res

    def test_bocpd_verified_when_column_present(self):
        tk = RegimeToolKit(
            cal_preds_df=_make_preds_df(include_bocpd=True),
            features_df=_make_feats_df(),
        )
        res = tk.get_changepoint_signal("2024-01-02")
        assert res["status"] == "VERIFIED"
        assert 0.0 <= res["bocpd_changepoint_probability"] <= 1.0

    def test_get_historical_analogs_risk_off(self, toolkit):
        result = toolkit.get_historical_analogs("Risk-Off")
        assert "historical_analogs" in result
        assert len(result["historical_analogs"]) >= 1

    def test_stale_governance_metrics_cannot_appear_in_toolkit(self, toolkit):
        gov = toolkit.get_governance_metadata()
        gov_str = json.dumps(gov)
        assert "0.0003" not in gov_str
        assert "0.0014" not in gov_str
        assert "0.0011" not in gov_str

    def test_get_backtest_context(self, toolkit):
        result = toolkit.get_backtest_context()
        assert "strategy_cagr" in result
        assert result["strategy_cagr"] == "12.67%"

    def test_build_evidence_bundle_has_all_keys(self, toolkit):
        bundle = toolkit.build_evidence_bundle("2024-01-02")
        for key in [
            "regime_state", "feature_snapshot", "shap_attributions",
            "changepoint_signal", "conformal_context", "historical_analogs",
            "governance", "backtest_context", "dominant_regime",
        ]:
            assert key in bundle, f"Missing key: {key}"

    def test_strict_source_status_contract(self, toolkit):
        allowed_statuses = {"VERIFIED", "UNAVAILABLE", "QUARANTINED", "SYNTHETIC_SCENARIO"}
        bundle = toolkit.build_evidence_bundle("2024-01-02")
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
            assert evidence_obj["status"] in allowed_statuses

    def test_empty_preds_returns_unavailable(self):
        tk = RegimeToolKit(cal_preds_df=pd.DataFrame(), features_df=pd.DataFrame())
        result = tk.get_regime_state("2024-01-02")
        assert result["status"] == "UNAVAILABLE"

    def test_invalid_date_returns_unavailable(self, toolkit):
        result = toolkit.get_regime_state("not-a-date")
        assert result["status"] == "UNAVAILABLE"


# ===========================================================================
# 3. ResponseValidator Programmatic Grounding Tests
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
        return tk.build_evidence_bundle("2024-01-02")

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
# 4. ScenarioParser Tests (General Shocks & USD/INR Sign Semantics)
# ===========================================================================

class TestScenarioParser:
    @pytest.fixture
    def parser(self):
        return ScenarioParser()

    def test_parse_vix_increase_percent(self, parser):
        result = parser.parse("Increase India VIX by 30%")
        assert result.is_valid
        shock = result.shocks[0]
        assert shock.feature_name == "vix_level"
        assert shock.shock_value > 0

    def test_parse_nifty_decrease_percent(self, parser):
        result = parser.parse("Drop NIFTY 5 percent")
        assert result.is_valid
        shock = result.shocks[0]
        assert shock.feature_name == "nifty_ret_1d"
        assert shock.shock_value < 0

    def test_parse_vix_absolute_spike(self, parser):
        result = parser.parse("Spike VIX to 45")
        assert result.is_valid
        shock = result.shocks[0]
        assert shock.feature_name == "vix_level"

    def test_parse_compound_shock(self, parser):
        result = parser.parse("Increase India VIX by 25% and drop NIFTY 4 percent")
        assert result.is_valid
        assert len(result.shocks) == 2
        features = [s.feature_name for s in result.shocks]
        assert "vix_level" in features
        assert "nifty_ret_1d" in features

    def test_vix_shock_direction_positive(self, parser):
        r = parser.parse("Increase VIX by 10 percent")
        assert r.is_valid
        assert r.shocks[0].shock_value > 0

    def test_vix_shock_direction_negative(self, parser):
        r = parser.parse("Decrease VIX by 20%")
        assert r.is_valid
        assert r.shocks[0].shock_value < 0

    def test_rupee_shock(self, parser):
        r = parser.parse("USD/INR appreciates 3 percent")
        assert r.is_valid
        assert r.shocks[0].feature_name == "usdinr_ret_21d"

    def test_invalid_input_no_shocks(self, parser):
        r = parser.parse("The market is doing something interesting today")
        assert not r.is_valid
        assert len(r.warnings) > 0

    def test_to_shock_vector_keys_are_feature_names(self, parser):
        r = parser.parse("Increase India VIX by 30%")
        sv = r.to_shock_vector()
        assert all(isinstance(k, str) for k in sv.keys())
        assert all(isinstance(v, float) for v in sv.values())

    def test_to_display_table_structure(self, parser):
        r = parser.parse("Increase India VIX by 30%")
        tbl = r.to_display_table()
        assert isinstance(tbl, list)
        assert len(tbl) >= 1
        assert "Feature" in tbl[0]
        assert "Shock" in tbl[0]

    def test_parse_method_is_regex(self, parser):
        r = parser.parse("Increase VIX by 20%")
        assert r.parse_method == "REGEX"

    def test_vix_shock_cap(self, parser):
        r = parser.parse("Increase VIX by 500 points")
        assert r.is_valid
        assert r.shocks[0].shock_value <= 60.0

    def test_preset_scenarios_parseable(self, parser):
        for preset in ScenarioParser.PRESET_SCENARIOS:
            r = parser.parse(preset["text"])
            assert r.is_valid, f"Preset failed to parse: {preset['label']}"

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

        assert val_usdinr > 0, "USD/INR appreciates should have positive delta"
        assert val_rupee < 0, "Rupee appreciates should have negative delta"
        assert val_usdinr == -val_rupee


# ===========================================================================
# 5. Scenario Lab Simulation Engine Labeling Tests
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
        assert "Monte Carlo" not in fallback_result.engine_used

    def test_baseline_provenance_distinguishes_verified_vs_synthetic(
        self, parsed_scenario, baseline_features
    ):
        """Simulation result must explicitly label baseline provenance."""
        # When real baseline probabilities are supplied
        real_probs = {"Risk-On": 0.05, "Late-Cycle": 0.05, "Transitional": 0.10, "Post-Shock": 0.80, "Risk-Off": 0.00}
        verified_res: ScenarioSimulationResult = simulate_scenario(
            parsed=parsed_scenario,
            baseline_features=baseline_features,
            baseline_probs=real_probs,
            baseline_provenance="VERIFIED MODEL OUTPUT",
        )
        assert verified_res.baseline_provenance == "VERIFIED MODEL OUTPUT"
        assert verified_res.baseline_regime == "Post-Shock"
        assert verified_res.baseline_prob == 0.80

        # When no baseline probabilities are supplied (synthetic demo fallback)
        synthetic_res: ScenarioSimulationResult = simulate_scenario(
            parsed=parsed_scenario,
            baseline_features=baseline_features,
            baseline_probs=None,
        )
        assert synthetic_res.baseline_provenance == "SYNTHETIC DEMO BASELINE"
        assert synthetic_res.baseline_regime == "Risk-On"
        assert synthetic_res.baseline_prob == 0.20


# ===========================================================================
# 6. RegimeCopilot Integration Tests
# ===========================================================================

class TestRegimeCopilot:
    @pytest.fixture
    def copilot(self):
        toolkit = RegimeToolKit(
            cal_preds_df=_make_preds_df(include_bocpd=False),
            features_df=_make_feats_df(),
        )
        return RegimeCopilot(toolkit=toolkit, provider=MockProvider(), log_interactions=False)

    def test_ask_without_date_returns_error(self, copilot):
        result = copilot.ask("What is the current regime?")
        assert "error" in result

    def test_set_date_returns_bundle(self, copilot):
        bundle = copilot.set_date("2024-01-02")
        assert "dominant_regime" in bundle
        assert "regime_state" in bundle

    def test_ask_after_set_date_returns_structured(self, copilot):
        copilot.set_date("2024-01-02")
        result = copilot.ask("What is the current regime?")
        assert "model_output" in result
        assert "calculated_evidence" in result
        assert "ai_interpretation" in result
        assert "provider" in result
        assert result["grounding_trusted"] is True

    def test_model_output_has_dominant_regime(self, copilot):
        copilot.set_date("2024-01-02")
        result = copilot.ask("Tell me about the regime.")
        mo = result["model_output"]
        assert "dominant_regime" in mo
        assert mo["dominant_regime"] in ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]

    def test_calculated_evidence_has_entropy(self, copilot):
        copilot.set_date("2024-01-02")
        result = copilot.ask("What is the uncertainty?")
        ce = result["calculated_evidence"]
        assert "predictive_entropy_nats" in ce
        assert isinstance(ce["predictive_entropy_nats"], float)
        assert ce["predictive_entropy_nats"] >= 0.0

    def test_clear_history_resets(self, copilot):
        copilot.set_date("2024-01-02")
        copilot.ask("Question one.")
        assert len(copilot.history) > 0
        copilot.clear_history()
        assert len(copilot.history) == 0

    def test_history_grows_with_turns(self, copilot):
        copilot.set_date("2024-01-02")
        copilot.ask("First question.")
        copilot.ask("Second question.")
        assert len(copilot.history) == 4

    def test_provider_name_in_result(self, copilot):
        copilot.set_date("2024-01-02")
        result = copilot.ask("What regime?")
        assert result["provider"] == copilot.provider_name

    def test_suggested_questions_nonempty(self, copilot):
        copilot.set_date("2024-01-02")
        qs = copilot.get_suggested_questions()
        assert isinstance(qs, list)
        assert len(qs) >= 1

    def test_date_change_resets_history(self, copilot):
        copilot.set_date("2024-01-02")
        copilot.ask("First question.")
        copilot.set_date("2024-01-03")
        assert len(copilot.history) == 0


# ===========================================================================
# 7. InteractionLogger Auditability Tests
# ===========================================================================

class TestInteractionLogger:
    def test_log_and_retrieve(self, tmp_path):
        logger = InteractionLogger(log_dir=tmp_path / "logs")
        logger.log(
            interaction_type="COPILOT",
            query="What regime?",
            response="Late-Cycle at 85%.",
            provider_name="MockProvider",
        )
        records = logger.get_recent(10)
        assert len(records) == 1
        assert records[0]["interaction_type"] == "COPILOT"
        assert records[0]["query"] == "What regime?"

    def test_session_count(self, tmp_path):
        logger = InteractionLogger(log_dir=tmp_path / "logs")
        for i in range(3):
            logger.log("COPILOT", f"q{i}", "r", "Mock")
        assert logger.get_session_count() == 3

    def test_log_writes_valid_json(self, tmp_path):
        logger = InteractionLogger(log_dir=tmp_path / "logs")
        logger.log("SCENARIO_LAB", "VIX +30%", "shock applied", "REGEX")
        log_files = list((tmp_path / "logs").glob("*.jsonl"))
        assert len(log_files) == 1
        with open(log_files[0]) as f:
            for line in f:
                record = json.loads(line)
                assert "timestamp" in record
                assert "interaction_type" in record

    def test_interaction_logger_records_all_eight_fields(self, tmp_path):
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

        assert "date" in rec and rec["date"] == "2024-06-04"
        assert "provider" in rec and rec["provider"] == "MockProvider"
        assert "model_name" in rec and rec["model_name"] == "deterministic-mock-v1"
        assert "user_query" in rec and rec["user_query"] == "What is the current regime?"
        assert "evidence_bundle_hash" in rec and len(rec["evidence_bundle_hash"]) == 16
        assert "response_hash" in rec and len(rec["response_hash"]) == 16
        assert "evidence_source_statuses" in rec and rec["evidence_source_statuses"] == source_statuses
        assert "scenario_id" in rec and rec["scenario_id"] == "SCEN-TEST-001"
