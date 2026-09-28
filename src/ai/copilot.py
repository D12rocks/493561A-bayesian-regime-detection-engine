"""
AI Regime Copilot.

A grounded natural-language analyst assistant for the RegimeLab platform.

DESIGN CONSTRAINTS:
1. The LLM NEVER independently generates regime probabilities.
2. All regime outputs come exclusively from the statistical engine (RegimeToolKit).
3. Every response clearly separates MODEL OUTPUT, CALCULATED EVIDENCE, AI INTERPRETATION.
4. All interactions are logged via InteractionLogger with evidence hash + grounding status.
5. No write access to any data artifact.
6. ResponseValidator checks all numerical claims before display.
7. UNAVAILABLE evidence is surfaced as unavailable — not zero, not representative.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.ai.llm_provider import LLMProvider, get_provider
from src.ai.tools import RegimeToolKit
from src.ai.interaction_logger import InteractionLogger
from src.ai.response_validator import ResponseValidator

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """
You are the AI Regime Copilot for RegimeLab, an institutional quantitative research platform
built by Zetheta Algorithms. You are a grounded market analyst assistant.

MANDATORY RULES — NEVER VIOLATE:
1. You MUST NOT independently generate or invent regime probabilities, model scores,
   backtest metrics, or any statistical outputs. Only use the structured evidence below.
2. Always structure your response with exactly these three sections:
   **MODEL OUTPUT**: Quote exact values from the evidence (regime, probabilities, metrics).
   **CALCULATED EVIDENCE**: Specific computed facts (entropy, SHAP, BOCPD).
   **AI INTERPRETATION**: Your analytical narrative. Mark clearly as interpretation, not fact.
3. If an evidence field has status=UNAVAILABLE:
   - Do NOT claim any numerical value for it.
   - State exactly: "The [field] is not available in the current evidence bundle."
   - Never treat UNAVAILABLE as zero, estimated, or representative.
4. NEVER claim a regime probability different from the evidence.
5. NEVER invent SHAP values, changepoint probabilities, or governance metrics.
6. If asked something beyond the evidence, say:
   "The statistical engine does not provide data for this query. I cannot speculate."
7. This is a confidential internal platform. Do not suggest external data sources.

STRUCTURED EVIDENCE (authoritative — status field indicates reliability):
```json
{evidence_json}
```

STATUS MEANINGS:
  VERIFIED   = loaded from real artifact or live DataFrames — quote these values exactly.
  UNAVAILABLE = data absent — do NOT claim any value.
  QUARANTINED = artifact exists but failed checks — treat as UNAVAILABLE.

Your role is to help the analyst understand and interpret verified evidence only.
"""


class RegimeCopilot:
    """
    Manages the AI Regime Copilot session.
    Maintains conversation history, injects evidence context, enforces grounding,
    and validates all LLM responses before display.
    """

    def __init__(
        self,
        toolkit: RegimeToolKit,
        provider: Optional[LLMProvider] = None,
        log_interactions: bool = True,
    ) -> None:
        self.toolkit = toolkit
        self.provider = provider or get_provider()
        self._audit_logger = InteractionLogger() if log_interactions else None
        self._validator = ResponseValidator()
        self._history: List[Dict[str, str]] = []
        self._current_date: Optional[str] = None
        self._evidence_bundle: Optional[Dict[str, Any]] = None

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        """Extract model identifier from provider."""
        name = self.provider.provider_name
        # e.g. "Ollama (llama3.2)" → "llama3.2"
        if "(" in name and ")" in name:
            return name[name.find("(") + 1: name.find(")")]
        return name

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def set_date(self, date_str: str) -> Dict[str, Any]:
        """
        Set the analysis date and pre-compute the evidence bundle.
        Resets conversation history for the new date context.
        """
        self._current_date = date_str
        self._evidence_bundle = self.toolkit.build_evidence_bundle(date_str)
        self._history = []
        return self._evidence_bundle

    def ask(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query and return a structured, validated response.

        Returns dict with:
            model_output       : dict (regime state from engine, always from toolkit)
            calculated_evidence: dict (specific metrics from toolkit)
            ai_interpretation  : str (LLM narrative, only if grounding_trusted)
            grounding_trusted  : bool
            violations         : list[str]
            warnings           : list[str]
            provider           : str
            evidence_bundle    : dict (full evidence for transparency)
            date               : str
        """
        if not self._current_date or not self._evidence_bundle:
            return {
                "error": "No date selected. Please set an analysis date first.",
                "ai_interpretation": "",
                "model_output": {},
                "calculated_evidence": {},
                "grounding_trusted": False,
                "violations": [],
                "warnings": [],
                "provider": self.provider_name,
            }

        evidence_json = json.dumps(self._evidence_bundle, indent=2, default=str)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(evidence_json=evidence_json)

        # Call LLM
        raw_response = self.provider.chat(
            system_prompt=system_prompt,
            user_message=user_query,
            history=self._history[-8:],
        )

        # Validate response against authoritative evidence
        validation = self._validator.validate(raw_response, self._evidence_bundle)
        display_response = validation.safe_response  # filtered if violations exist

        # Update conversation history with the DISPLAYED (possibly safe) response
        self._history.append({"role": "user", "content": user_query})
        self._history.append({"role": "assistant", "content": display_response})

        # Structured model output — always from toolkit, never from LLM
        regime_state = self._evidence_bundle.get("regime_state", {})
        shap = self._evidence_bundle.get("shap_attributions", {})
        cp = self._evidence_bundle.get("changepoint_signal", {})

        model_output: Dict[str, Any] = {}
        if regime_state.get("status") == "VERIFIED":
            model_output = {
                "dominant_regime": regime_state.get("dominant_regime"),
                "dominant_probability": f"{regime_state.get('dominant_prob', 0)*100:.1f}%",
                "regime_probabilities": regime_state.get("regime_probabilities", {}),
                "conformal_set": regime_state.get("conformal_set", []),
                "status": "VERIFIED",
            }
        else:
            model_output = {
                "status": "UNAVAILABLE",
                "reason": regime_state.get("reason", "Regime state not available."),
            }

        calculated_evidence: Dict[str, Any] = {}
        # Entropy
        if regime_state.get("status") == "VERIFIED":
            calculated_evidence["predictive_entropy_nats"] = round(
                regime_state.get("predictive_entropy", 0.0), 4
            )
            calculated_evidence["conformal_set_size"] = regime_state.get("conformal_set_size", 0)
        else:
            calculated_evidence["predictive_entropy_nats"] = "UNAVAILABLE"
            calculated_evidence["conformal_set_size"] = "UNAVAILABLE"

        # BOCPD
        if cp.get("status") == "VERIFIED":
            calculated_evidence["bocpd_changepoint_prob"] = (
                f"{cp.get('bocpd_changepoint_probability', 0)*100:.1f}%"
            )
            calculated_evidence["changepoint_alert"] = cp.get("alert_level", "UNKNOWN")
        else:
            calculated_evidence["bocpd_changepoint_prob"] = "UNAVAILABLE"
            calculated_evidence["changepoint_alert"] = "UNAVAILABLE"

        # SHAP
        if shap.get("status") == "VERIFIED":
            calculated_evidence["top_shap_drivers"] = shap.get("top_shap_features", [])[:3]
        else:
            calculated_evidence["top_shap_drivers"] = "UNAVAILABLE"

        # Evidence source statuses for logging
        source_statuses = self.toolkit.get_evidence_source_statuses(self._evidence_bundle)

        # Audit log
        if self._audit_logger:
            self._audit_logger.log(
                interaction_type="COPILOT",
                query=user_query,
                response=display_response,
                provider_name=self.provider_name,
                model_name=self.model_name,
                evidence_bundle=self._evidence_bundle,
                evidence_source_statuses=source_statuses,
                grounding_trusted=validation.is_trusted,
                violations=validation.violations,
                analysis_date=self._current_date,
            )

        return {
            "model_output": model_output,
            "calculated_evidence": calculated_evidence,
            "ai_interpretation": display_response,
            "grounding_trusted": validation.is_trusted,
            "violations": validation.violations,
            "warnings": validation.warnings,
            "provider": self.provider_name,
            "evidence_bundle": self._evidence_bundle,
            "date": self._current_date,
        }

    def clear_history(self) -> None:
        """Reset conversation history (preserves current date/evidence)."""
        self._history = []

    def get_suggested_questions(self) -> List[str]:
        """Return context-appropriate starter questions."""
        if not self._evidence_bundle:
            return [
                "What is the current regime?",
                "What are the main drivers?",
                "Is a regime transition forming?",
            ]

        regime = self._evidence_bundle.get("dominant_regime") or "the current state"
        cp_prob = self._evidence_bundle.get("changepoint_probability")
        entropy = self._evidence_bundle.get("predictive_entropy")
        shap_status = self._evidence_bundle.get("shap_attributions", {}).get("status")

        questions = [f"Why is the engine calling {regime} right now?"]

        if cp_prob is not None and cp_prob > 0.25:
            questions.append("What evidence supports a regime transition?")
        elif cp_prob is None:
            questions.append("Is BOCPD changepoint signal available for this date?")

        if entropy is not None and entropy > 0.4:
            questions.append("Why is the model uncertain about the current regime?")

        if shap_status == "VERIFIED":
            questions.append("What are the top SHAP drivers for this regime call?")
        else:
            questions.append("Why are SHAP attributions unavailable?")

        questions.extend([
            "What does the conformal prediction set tell us?",
            "Are there historical analogs for this regime?",
            "How does the backtest perform during this type of regime?",
        ])
        return questions[:5]
