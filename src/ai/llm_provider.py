"""
LLM Provider Abstraction Layer.

Implements a pluggable interface for LLM backends.
Default: Ollama (local, no external transmission of confidential data).
Fallback: MockProvider — deterministic template-based analyst.

CONFIDENTIALITY: No confidential project data is ever transmitted externally.
All inference runs locally via Ollama (localhost:11434) or the mock engine.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2"  # Tested with Ollama 0.3+
_TIMEOUT_S = 60


class LLMProvider(ABC):
    """Abstract base class for all LLM backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is reachable and ready."""

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Send a chat message and return the assistant reply.

        Args:
            system_prompt: Grounding instructions injected as the system role.
            user_message: User query.
            history: Prior conversation turns [{"role": "user/assistant", "content": ...}].

        Returns:
            String reply from the assistant.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""


# ---------------------------------------------------------------------------
# Ollama Provider (primary)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """
    Calls a locally-running Ollama server.
    All data stays on-premise — no external network transmission.
    """

    def __init__(self, model: str = _DEFAULT_MODEL, base_url: str = _OLLAMA_BASE_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return f"Ollama ({self.model})"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return False
            # Check model is pulled
            tags = r.json().get("models", [])
            names = [m.get("name", "").split(":")[0] for m in tags]
            return self.model.split(":")[0] in names
        except Exception:
            return False

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,   # Low temperature for analytical grounding
                "top_p": 0.9,
                "num_predict": 800,
            },
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=_TIMEOUT_S,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Ollama call failed: %s", exc)
            return f"[Ollama unavailable: {exc}]"


# ---------------------------------------------------------------------------
# Mock / Deterministic Fallback Provider
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """
    Template-based analyst that synthesises a grounded interpretation
    from structured evidence without any LLM call.

    Used when Ollama is not installed or the model is not pulled.
    Responses are deterministic and factual — no hallucinations.
    """

    @property
    def provider_name(self) -> str:
        return "Deterministic Fallback Analyst"

    def is_available(self) -> bool:
        return True  # Always available

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Parse the structured evidence block embedded in system_prompt
        and produce a factual template response.
        """
        q_lower = user_message.lower()

        # Extract JSON evidence if present
        evidence: Dict[str, Any] = {}
        try:
            start = system_prompt.find("```json")
            if start != -1:
                end = system_prompt.find("```", start + 6)
                evidence = json.loads(system_prompt[start + 7 : end])
        except Exception:
            pass

        regime = evidence.get("dominant_regime")
        prob = evidence.get("dominant_prob")
        entropy = evidence.get("predictive_entropy")
        cp_prob = evidence.get("changepoint_probability")
        conf_set = evidence.get("conformal_set", [])
        top_features = evidence.get("top_shap_features", [])

        if any(w in q_lower for w in ["regime", "state", "current"]):
            if regime is None or prob is None:
                return (
                    "The regime state is UNAVAILABLE in the current evidence bundle. "
                    "No probability has been estimated or substituted. "
                    "*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
                )
            feat_str = (
                ", ".join(top_features[:3])
                if top_features
                else "SHAP attributions UNAVAILABLE"
            )
            entropy_str = f"{entropy:.3f} nats" if entropy is not None else "UNAVAILABLE"
            ambiguity_note = (
                "high ambiguity — multiple regimes are plausible"
                if (entropy is not None and entropy > 0.5)
                else "moderate confidence in the current state"
            )
            return (
                f"The current dominant regime is **{regime}** with a posterior probability of "
                f"**{prob*100:.1f}%**. Top feature drivers: {feat_str}. "
                f"Predictive entropy is {entropy_str}, indicating {ambiguity_note}. "
                f"*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
            )
        elif any(w in q_lower for w in ["changepoint", "transition", "switch"]):
            if cp_prob is None:
                return (
                    "The Bayesian Online Changepoint Detector signal is UNAVAILABLE for this date. "
                    "No changepoint probability is available in the authoritative predictions. "
                    "*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
                )
            return (
                f"The Bayesian Online Changepoint Detector reports a changepoint probability of "
                f"**{cp_prob*100:.1f}%**. "
                + (
                    "⚠️ This exceeds the 30% alerting threshold — a regime transition is actively forming."
                    if cp_prob > 0.30
                    else "Probability is below the 30% alerting threshold; regime continuity is the base case."
                )
                + " *(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
            )
        elif any(w in q_lower for w in ["conformal", "uncertainty", "set", "predict"]):
            if not conf_set:
                return (
                    "The conformal prediction set is UNAVAILABLE in the current evidence bundle. "
                    "*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
                )
            set_str = ", ".join(conf_set)
            entropy_str = f"{entropy:.3f} nats" if entropy is not None else "UNAVAILABLE"
            return (
                f"The 90% conformal prediction set is **{{{set_str}}}** (size {len(conf_set)}). "
                f"A set of size 1 indicates high certainty; size > 2 signals distributional ambiguity. "
                f"Total predictive entropy: {entropy_str}. "
                f"*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
            )
        elif any(w in q_lower for w in ["shap", "driver", "feature", "why", "reason"]):
            if not top_features:
                return (
                    "SHAP feature attributions are UNAVAILABLE for this date. "
                    "No representative SHAP values are substituted. "
                    "*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
                )
            feat_str = ", ".join(top_features)
            return (
                f"The top SHAP feature drivers for the current regime call are: **{feat_str}**. "
                f"SHAP values measure marginal contribution of each feature to the log-odds "
                f"of the dominant class relative to the base rate. "
                f"*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
            )
        else:
            reg_display = (
                f"{regime} @ {prob*100:.1f}%"
                if (regime and prob is not None)
                else "UNAVAILABLE"
            )
            cp_display = (
                f"{cp_prob*100:.1f}%"
                if cp_prob is not None
                else "UNAVAILABLE"
            )
            return (
                f"I can answer questions about the current regime ({reg_display}), "
                f"changepoint probability ({cp_display}), conformal sets, SHAP drivers, "
                f"uncertainty, and historical analogs. "
                f"*(Deterministic fallback — install Ollama for LLM-enhanced analysis.)*"
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(model: str = _DEFAULT_MODEL, prefer_ollama: bool = True) -> LLMProvider:
    """
    Return the best available provider.
    Tries Ollama first; falls back to MockProvider gracefully.
    """
    if prefer_ollama:
        provider = OllamaProvider(model=model)
        if provider.is_available():
            logger.info("AI Provider: Ollama (%s) — local inference active.", model)
            return provider
        logger.info("Ollama unavailable or model not pulled. Using deterministic fallback.")
    return MockProvider()
