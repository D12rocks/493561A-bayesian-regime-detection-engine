"""
Natural-Language Scenario Parser for the AI Stress Lab.

Converts natural-language shock descriptions into quantified feature shock vectors
that can be injected into the existing Monte Carlo / ScenarioEngine infrastructure.

Design:
- Primary: regex-based pattern matching (no LLM required, deterministic)
- Secondary: LLM-assisted disambiguation for complex multi-part instructions
- Always presents parsed scenario to user for confirmation before running
- SYNTHETIC flag explicitly set on all parsed shocks

USD/INR SIGN SEMANTICS (CRITICAL):
    usdinr_ret_21d is the 21-day RETURN of the USD/INR exchange rate.
    - usdinr_ret_21d > 0 means USD/INR rose  → Rupee WEAKENED / Depreciated
    - usdinr_ret_21d < 0 means USD/INR fell  → Rupee STRENGTHENED / Appreciated

    Natural language disambiguation:
    - "USD/INR increases/rises/appreciates"    → usdinr_ret_21d delta > 0  (USD up vs INR)
    - "USD/INR decreases/falls/depreciates"    → usdinr_ret_21d delta < 0  (USD down vs INR)
    - "USD/INR +3%"                            → usdinr_ret_21d delta = +0.03 (Rupee weakens)
    - "USD/INR -3%"                            → usdinr_ret_21d delta = -0.03 (Rupee strengthens)
    - "Rupee appreciates/strengthens/rises"    → usdinr_ret_21d delta < 0  (INR up = USD/INR down)
    - "Rupee depreciates/weakens/falls"        → usdinr_ret_21d delta > 0  (INR down = USD/INR up)

    The subject of the clause determines whether the linguistic direction is inverted.
    "INR" / "rupee" subjects → INVERT direction.
    "USD/INR" / "dollar" / "usd" subjects → do NOT invert.

Supported feature targets:
    India VIX / VIX                → vix_level (absolute add or % change)
    NIFTY return / market return   → nifty_ret_1d (percentage points)
    Realized volatility / vol      → nifty_vol_ewma_21d
    USD/INR / Rupee                → usdinr_ret_21d
    Midcap / breadth               → breadth_midcap_ret_21d
    Distance SMA / trend           → nifty_dist_sma50
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParsedShock:
    """A single parsed feature shock."""
    feature_name: str         # Internal column name
    display_name: str         # Human-readable label
    shock_type: str           # "additive" or "multiplicative"
    shock_value: float        # The delta to apply (in native units)
    original_text: str        # Matched text from user input
    confidence: float = 1.0  # 0–1 parsing confidence


@dataclass
class ParsedScenario:
    """Full parsed shock scenario ready for confirmation + simulation."""
    natural_language_input: str
    shocks: List[ParsedShock] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scenario_type: str = "SYNTHETIC_NL_STRESS"
    parse_method: str = "REGEX"

    def to_shock_vector(self) -> Dict[str, float]:
        """Convert to the shock_vector format expected by ScenarioEngine."""
        return {s.feature_name: s.shock_value for s in self.shocks}

    def to_display_table(self) -> List[Dict[str, str]]:
        """Return a list of dicts for Streamlit table rendering."""
        rows = []
        for s in self.shocks:
            sign = "+" if s.shock_value >= 0 else ""
            unit = "pts" if s.shock_type == "additive" else "%"
            rows.append({
                "Feature": s.display_name,
                "Shock": f"{sign}{s.shock_value:.4f} {unit}",
                "Type": s.shock_type.upper(),
                "Confidence": f"{s.confidence*100:.0f}%",
            })
        return rows

    @property
    def is_valid(self) -> bool:
        return len(self.shocks) > 0


@dataclass
class ScenarioSimulationResult:
    """Result of running a parsed scenario through simulation."""
    scenario_type: str = "SYNTHETIC_NL_STRESS"
    ui_label: str = "HYPOTHETICAL SCENARIO — NOT A FORECAST"
    engine_used: str = "FULL MONTE CARLO"  # "FULL MONTE CARLO" or "PARAMETRIC FALLBACK"
    is_monte_carlo: bool = True
    status: str = "SYNTHETIC_SCENARIO"
    baseline_regime: str = ""
    baseline_prob: float = 0.0
    stressed_dominant_regime: str = ""
    stressed_dominant_prob: float = 0.0
    stressed_regime_probabilities: Dict[str, float] = field(default_factory=dict)
    stressed_entropy: float = 0.0
    shock_vector: Dict[str, float] = field(default_factory=dict)
    var_95: float = 0.0
    cvar_95: float = 0.0
    median_return: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Feature mapping: natural language terms → internal column names
# ---------------------------------------------------------------------------

_FEATURE_MAP: List[Tuple[List[str], str, str]] = [
    (
        ["india vix", "vix level", "vix", "volatility index", "implied volatility"],
        "vix_level",
        "India VIX Level",
    ),
    (
        ["nifty return", "nifty 1d return", "nifty", "market return", "equity return", "nifty50", "nifty 50"],
        "nifty_ret_1d",
        "NIFTY 1-Day Return",
    ),
    (
        ["realized volatility", "realized vol", "ewma vol", "historical volatility", "hv"],
        "nifty_vol_ewma_21d",
        "21D EWMA Realized Volatility",
    ),
    (
        ["midcap breadth", "midcap", "breadth", "small cap", "smallcap", "breadth spread"],
        "breadth_midcap_ret_21d",
        "Midcap Breadth Spread",
    ),
    (
        ["sma distance", "sma", "trend", "distance sma", "moving average distance"],
        "nifty_dist_sma50",
        "NIFTY Distance from 50D SMA",
    ),
]

# USD/INR aliases: linguistic direction = usdinr_ret_21d direction (no inversion)
_FX_USDINR_ALIASES = ["usd/inr", "usd inr", "dollar rupee", "dollar-rupee", "fx rate", "dollar"]

# Rupee/INR aliases: linguistic direction is INVERTED for usdinr_ret_21d
# (rupee goes up → USD/INR goes down → usdinr_ret_21d delta is NEGATIVE)
_FX_RUPEE_ALIASES = ["rupee", "inr", "indian rupee"]

# ---------------------------------------------------------------------------
# Direction words
# ---------------------------------------------------------------------------

_INCREASE_WORDS = r"(increas|ris|spike|jump|surge|hike|up\b|boost|elevat|add|grow|appreciat|strengthen)"
_DECREASE_WORDS = r"(decreas|fall|drop|crash|plunge|cut|down\b|reduc|lower|shrink|depreciat|compress|weaken)"


def _find_feature(text: str) -> Optional[Tuple[str, str, bool]]:
    """
    Find the best-matching feature column from text.
    Returns (column_name, display_name, invert_direction).
    invert_direction=True for Rupee-subject FX clauses.
    """
    text_lower = text.lower()

    # Check USD/INR aliases first (no inversion)
    for alias in _FX_USDINR_ALIASES:
        if alias in text_lower:
            return "usdinr_ret_21d", "USD/INR 21D Return", False

    # Check Rupee aliases second (inversion)
    for alias in _FX_RUPEE_ALIASES:
        if alias in text_lower:
            return "usdinr_ret_21d", "USD/INR 21D Return (Rupee Basis — direction inverted)", True

    for aliases, col, display in _FEATURE_MAP:
        for alias in aliases:
            if alias in text_lower:
                return col, display, False

    return None


def _extract_magnitude(text: str) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Extract numeric magnitude, unit, and explicit sign (+ or -) from text.
    Returns (magnitude_abs, unit_type, explicit_sign).
    """
    patterns = [
        (r"([+-]?\s*\d+\.?\d*)\s*%", "percent"),
        (r"([+-]?\s*\d+\.?\d*)\s*percent", "percent"),
        (r"([+-]?\s*\d+\.?\d*)\s*bps?\b", "bps"),
        (r"([+-]?\s*\d+\.?\d*)\s*basis\s*points?", "bps"),
        (r"([+-]?\s*\d+\.?\d*)\s*pts?\b", "points"),
        (r"([+-]?\s*\d+\.?\d*)\s*points?", "points"),
        (r"to\s+(\d+\.?\d*)", "absolute"),
        (r"([+-]?\s*\d+\.?\d*)", "absolute"),
    ]
    for pattern, unit in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw_str = m.group(1).replace(" ", "")
            explicit_sign = None
            if raw_str.startswith("+"):
                explicit_sign = "+"
                val = float(raw_str[1:])
            elif raw_str.startswith("-"):
                explicit_sign = "-"
                val = float(raw_str[1:])
            else:
                val = float(raw_str)
            return val, unit, explicit_sign

    return None, "unknown", None


def _compute_shock_value(
    feature_col: str,
    direction: str,
    magnitude: float,
    unit: str,
    current_value: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Convert magnitude + unit → native feature delta.
    Returns (delta, shock_type).
    """
    sign = 1.0 if direction == "increase" else -1.0

    if feature_col == "vix_level":
        if unit == "percent":
            baseline = current_value if current_value else 15.0
            delta = sign * baseline * magnitude / 100.0
        elif unit == "absolute":
            delta = magnitude - (current_value or 15.0)
        else:
            delta = sign * magnitude
        return delta, "additive"

    elif feature_col in ("nifty_ret_1d", "nifty_vol_ewma_21d", "usdinr_ret_21d",
                          "breadth_midcap_ret_21d", "nifty_dist_sma50"):
        if unit == "percent":
            delta = sign * magnitude / 100.0
        elif unit == "bps":
            delta = sign * magnitude / 10000.0
        else:
            delta = sign * magnitude / 100.0
        return delta, "additive"

    return sign * magnitude, "additive"


class ScenarioParser:
    """
    Parses natural-language shock descriptions into structured ParsedScenario objects.
    """

    def __init__(
        self,
        llm_provider=None,
        interaction_logger=None,
    ) -> None:
        self._llm = llm_provider
        self._logger = interaction_logger

    def parse(
        self,
        text: str,
        current_features: Optional[Dict[str, float]] = None,
    ) -> ParsedScenario:
        """Parse natural-language scenario description."""
        scenario = ParsedScenario(natural_language_input=text)
        text_clean = text.strip()

        parts = re.split(r"\band\b|,|;", text_clean, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]

        parsed_any = False
        for part in parts:
            shock = self._parse_single_shock(part, current_features)
            if shock:
                scenario.shocks.append(shock)
                parsed_any = True
            else:
                scenario.warnings.append(
                    f"Could not parse: '{part}'. Please specify the feature and magnitude clearly."
                )

        if not parsed_any and self._llm and self._llm.is_available():
            scenario = self._llm_disambiguate(text, scenario, current_features)
            scenario.parse_method = "LLM_ASSISTED"

        if not scenario.is_valid:
            scenario.warnings.append(
                "No valid shock vectors could be extracted. "
                "Try: 'Increase India VIX by 30%' or 'USD/INR +3%'."
            )

        if self._logger:
            self._logger.log(
                interaction_type="SCENARIO_LAB",
                query=text,
                response=str(scenario.to_shock_vector()),
                provider_name=scenario.parse_method,
                evidence_bundle={"shocks_parsed": len(scenario.shocks)},
                extra={"warnings": scenario.warnings},
            )

        return scenario

    def _parse_single_shock(
        self,
        text: str,
        current_features: Optional[Dict[str, float]] = None,
    ) -> Optional[ParsedShock]:
        """Parse a single shock clause."""
        text_lower = text.lower()

        # 1. Identify feature (returns invert_direction for FX)
        feat_match = _find_feature(text_lower)
        if not feat_match:
            return None
        col, display, invert_direction = feat_match

        # 2. Extract magnitude & sign if present
        magnitude, unit, explicit_sign = _extract_magnitude(text)

        # Default magnitude for standalone directional statements like "rupee appreciation"
        if magnitude is None:
            if any(w in text_lower for w in ["appreciat", "depreciat", "surge", "crash", "spike"]):
                magnitude = 3.0
                unit = "percent"
            else:
                return None

        # 3. Identify linguistic direction
        is_increase = False
        is_decrease = False

        if explicit_sign == "+":
            is_increase = True
        elif explicit_sign == "-":
            is_decrease = True
        else:
            is_increase = bool(re.search(_INCREASE_WORDS, text_lower, re.IGNORECASE))
            is_decrease = bool(re.search(_DECREASE_WORDS, text_lower, re.IGNORECASE))

            is_absolute = bool(re.search(r"\bto\s+\d", text_lower))
            if is_absolute:
                is_increase = True

            if not is_increase and not is_decrease:
                if "+" in text:
                    is_increase = True
                elif "-" in text:
                    is_decrease = True
                else:
                    is_increase = True

        # 4. Apply Rupee-subject direction inversion
        #    "Rupee appreciates" → linguistic is_increase=True → invert_direction=True
        #    → actual direction for usdinr_ret_21d = DECREASE (USD/INR falls)
        if invert_direction:
            is_increase, is_decrease = is_decrease, is_increase

        direction = "increase" if is_increase else "decrease"

        # 5. Get current value if available
        current_val = current_features.get(col) if current_features else None

        # 6. Compute shock delta
        shock_value, shock_type = _compute_shock_value(col, direction, magnitude, unit, current_val)

        # Sanity bounds
        if col == "vix_level" and abs(shock_value) > 60:
            shock_value = 60.0 * (1 if shock_value >= 0 else -1)

        return ParsedShock(
            feature_name=col,
            display_name=display,
            shock_type=shock_type,
            shock_value=round(shock_value, 6),
            original_text=text,
            confidence=0.85 if unit != "absolute" else 0.70,
        )

    def _llm_disambiguate(
        self,
        text: str,
        scenario: ParsedScenario,
        current_features: Optional[Dict[str, float]] = None,
    ) -> ParsedScenario:
        """LLM-assisted parsing for ambiguous inputs."""
        if not self._llm:
            return scenario

        prompt = (
            "You are a financial scenario parser. Convert the following natural-language shock "
            "description into a JSON list of shocks. Only output valid JSON, nothing else.\n"
            "Each shock must have: feature (one of: vix_level, nifty_ret_1d, nifty_vol_ewma_21d, "
            "usdinr_ret_21d, breadth_midcap_ret_21d, nifty_dist_sma50), "
            "direction (increase/decrease), magnitude (float), unit (percent/points/absolute).\n"
            "IMPORTANT for usdinr_ret_21d: direction=increase means USD/INR rises (Rupee weakens).\n"
            "Example: [{\"feature\": \"vix_level\", \"direction\": \"increase\", \"magnitude\": 30, \"unit\": \"percent\"}]\n"
        )
        try:
            import json as _json
            response = self._llm.chat(system_prompt=prompt, user_message=f"Parse: {text}")
            m = re.search(r"\[.*?\]", response, re.DOTALL)
            if not m:
                return scenario
            parsed = _json.loads(m.group(0))

            for item in parsed:
                col = item.get("feature")
                direction = item.get("direction", "increase")
                magnitude = float(item.get("magnitude", 0))
                unit = item.get("unit", "percent")

                display = col
                for aliases, c, d, *_ in (*_FEATURE_MAP, ([], "usdinr_ret_21d", "USD/INR 21D Return")):
                    if c == col:
                        display = d
                        break

                current_val = current_features.get(col) if current_features else None
                shock_value, shock_type = _compute_shock_value(col, direction, magnitude, unit, current_val)

                scenario.shocks.append(ParsedShock(
                    feature_name=col,
                    display_name=display,
                    shock_type=shock_type,
                    shock_value=round(shock_value, 6),
                    original_text=text,
                    confidence=0.65,
                ))
        except Exception as exc:
            logger.warning("LLM disambiguation failed: %s", exc)

        return scenario

    # ------------------------------------------------------------------
    # Preset scenarios
    # ------------------------------------------------------------------

    PRESET_SCENARIOS: List[Dict[str, Any]] = [
        {
            "label": "India VIX +30% (Volatility Spike)",
            "text": "Increase India VIX by 30%",
            "description": "Moderate volatility spike, mimicking early-stage risk-off",
        },
        {
            "label": "VIX Spike to 45 (Extreme Stress)",
            "text": "Spike VIX to 45",
            "description": "Severe tail-risk event, approaching March 2020 levels",
        },
        {
            "label": "NIFTY -5% Single-Day Crash",
            "text": "Drop NIFTY 5 percent",
            "description": "Equivalent to a 2-sigma negative day",
        },
        {
            "label": "USD/INR +3% (Rupee Depreciation Shock)",
            "text": "USD/INR +3%",
            "description": "Rupee depreciation — USD rises vs INR (usdinr_ret_21d +3%)",
        },
        {
            "label": "USD/INR -3% (Rupee Appreciation Shock)",
            "text": "USD/INR -3%",
            "description": "Rupee strengthening — USD falls vs INR (usdinr_ret_21d -3%)",
        },
        {
            "label": "Rupee Appreciation (INR Strengthening)",
            "text": "Rupee appreciation",
            "description": "Rupee strengthening — USD/INR falls (usdinr_ret_21d -3%)",
        },
        {
            "label": "Rupee Depreciation (INR Weakening)",
            "text": "Rupee depreciation",
            "description": "Rupee weakening — USD/INR rises (usdinr_ret_21d +3%)",
        },
        {
            "label": "Combined Stress: VIX +25%, NIFTY -4%",
            "text": "Increase India VIX by 25% and drop NIFTY 4 percent",
            "description": "Compound shock: vol spike + equity sell-off",
        },
        {
            "label": "Risk-On Recovery: VIX -20%, NIFTY +3%",
            "text": "Decrease India VIX by 20% and increase NIFTY 3 percent",
            "description": "Bullish recovery following risk-off episode",
        },
    ]


# ---------------------------------------------------------------------------
# Scenario Simulation Execution
# ---------------------------------------------------------------------------

def simulate_scenario(
    parsed: ParsedScenario,
    baseline_features: Dict[str, float],
    baseline_probs: Optional[Dict[str, float]] = None,
    force_parametric: bool = False,
) -> ScenarioSimulationResult:
    """
    Execute stress simulation with strict distinction between FULL MONTE CARLO
    and PARAMETRIC FALLBACK. Never presents parametric fallback as Monte Carlo.
    """
    shock_vec = parsed.to_shock_vector()
    shocked_features = dict(baseline_features)
    for feat, delta in shock_vec.items():
        if feat in shocked_features:
            if feat == "vix_level":
                shocked_features[feat] = max(8.0, shocked_features[feat] + delta)
            else:
                shocked_features[feat] = shocked_features[feat] + delta
        else:
            shocked_features[feat] = delta

    regime_names_ordered = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
    if baseline_probs:
        bl_probs_arr = np.array([baseline_probs.get(r, 0.2) for r in regime_names_ordered])
    else:
        bl_probs_arr = np.ones(5) / 5.0
    bl_probs_arr /= bl_probs_arr.sum()

    bl_dom_idx = int(np.argmax(bl_probs_arr))
    bl_regime = regime_names_ordered[bl_dom_idx]
    bl_prob = float(bl_probs_arr[bl_dom_idx])

    # Direction-aware heuristic regime shift for synthetic stress
    shock_severity = sum(abs(v) for v in shock_vec.values())
    vix_delta = shock_vec.get("vix_level", 0.0)
    nifty_delta = shock_vec.get("nifty_ret_1d", 0.0)

    shift = np.zeros(5)
    if vix_delta > 2 or nifty_delta < -0.02:
        strength = min(shock_severity * 0.15, 0.45)
        shift[4] += strength * 0.6  # Risk-Off
        shift[3] += strength * 0.3  # Post-Shock
        shift[2] += strength * 0.1  # Transitional
        shift[0] -= strength * 0.7  # Risk-On
        shift[1] -= strength * 0.3  # Late-Cycle
    elif vix_delta < -2 or nifty_delta > 0.02:
        strength = min(shock_severity * 0.15, 0.35)
        shift[0] += strength * 0.7  # Risk-On
        shift[1] += strength * 0.2  # Late-Cycle
        shift[4] -= strength * 0.6  # Risk-Off
        shift[3] -= strength * 0.3  # Post-Shock
        shift[2] -= strength * 0.1  # Transitional

    stressed_probs = np.clip(bl_probs_arr + shift, 0.001, 1.0)
    stressed_probs /= stressed_probs.sum()

    s_dom_idx = int(np.argmax(stressed_probs))
    s_dom_regime = regime_names_ordered[s_dom_idx]
    s_dom_prob = float(stressed_probs[s_dom_idx])
    s_entropy = float(-sum(p * np.log(max(p, 1e-12)) for p in stressed_probs))
    stressed_dict = dict(zip(regime_names_ordered, stressed_probs.tolist()))

    engine_used = "PARAMETRIC FALLBACK"
    is_mc = False
    var_95 = 0.0
    cvar_95 = 0.0
    median_ret = 0.0
    notes = ""

    if not force_parametric:
        try:
            from src.simulation.monte_carlo import MonteCarloRiskEngine
            from src.models.contracts import RegimeProbabilities
            p_obj = RegimeProbabilities(*stressed_probs)
            mc_engine = MonteCarloRiskEngine()
            paths = mc_engine.simulate_paths(
                current_probs=p_obj,
                horizon_days=21,
                n_paths=5000,
                random_seed=42,
            )
            tail = mc_engine.compute_tail_risk(paths, horizon_days=21)
            engine_used = "FULL MONTE CARLO"
            is_mc = True
            var_95 = float(tail.var_95)
            cvar_95 = float(tail.cvar_95)
            median_ret = float(tail.expected_return)
            notes = "Simulated via 5,000 fat-tailed Student-t Markov regime paths over 21-day horizon."
        except Exception as mc_err:
            engine_used = "PARAMETRIC FALLBACK"
            is_mc = False
            notes = f"Monte Carlo unavailable ({mc_err}); computed via parametric volatility."

    if not is_mc:
        vol_shocked = shocked_features.get("nifty_vol_ewma_21d", 0.15)
        var_95 = float(-1.645 * vol_shocked * np.sqrt(21 / 252))
        cvar_95 = float(var_95 * 1.25)
        median_ret = float(shocked_features.get("nifty_ret_1d", 0.0) * 21)
        if not notes:
            notes = "Parametric VaR fallback (normal distribution assumption, delta-normal approximation)."

    return ScenarioSimulationResult(
        scenario_type="SYNTHETIC_NL_STRESS",
        ui_label="HYPOTHETICAL SCENARIO — NOT A FORECAST",
        engine_used=engine_used,
        is_monte_carlo=is_mc,
        status="SYNTHETIC_SCENARIO",
        baseline_regime=bl_regime,
        baseline_prob=bl_prob,
        stressed_dominant_regime=s_dom_regime,
        stressed_dominant_prob=s_dom_prob,
        stressed_regime_probabilities=stressed_dict,
        stressed_entropy=s_entropy,
        shock_vector=shock_vec,
        var_95=var_95,
        cvar_95=cvar_95,
        median_return=median_ret,
        notes=notes,
    )
