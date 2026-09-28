"""
AI Layer for RegimeLab.

Provides:
- LLM provider abstraction (Ollama default, deterministic fallback)
- Read-only grounding tools for the AI Regime Copilot
- LLM response validator (grounding integrity checker)
- Natural-language scenario parser for the Stress Lab
- Interaction audit logger
"""

from src.ai.llm_provider import LLMProvider, OllamaProvider, MockProvider, get_provider
from src.ai.tools import RegimeToolKit
from src.ai.response_validator import ResponseValidator
from src.ai.copilot import RegimeCopilot
from src.ai.scenario_parser import ScenarioParser
from src.ai.interaction_logger import InteractionLogger

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "MockProvider",
    "get_provider",
    "RegimeToolKit",
    "ResponseValidator",
    "RegimeCopilot",
    "ScenarioParser",
    "InteractionLogger",
]
