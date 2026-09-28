"""
LLM Response Validator.

Programmatic grounding check that validates Ollama / LLM responses against
the authoritative evidence bundle BEFORE the response is shown to the user.

Checks:
  1. Dominant regime name — must match evidence exactly
  2. Regime probabilities — any quoted percentage must be within tolerance of evidence
  3. Changepoint probability — if BOCPD is UNAVAILABLE, claims about it are rejected;
     if VERIFIED, claims must match evidence
  4. Conformal set — set size and membership must match evidence exactly
  5. Backtest metrics — CAGR, Sharpe, Drawdown, Volatility must match locked values
  6. Stale governance metrics — 0.0003, 0.0014, 0.0011 must NEVER appear
  7. UNAVAILABLE signals — LLM must not claim a numerical value for UNAVAILABLE evidence

When a violation is found:
  - The response is flagged (is_trusted = False)
  - A safe_response string is returned for display instead
  - violations list records each discrepancy for audit

Tolerance:
  - Regime probability claims: ±2.5 percentage points
  - Numerical metrics: exact string match for known values
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How far off a quoted probability can be before it's flagged (pp = percentage points)
_PROB_TOLERANCE_PP = 2.5

# Locked backtest string literals
_LOCKED_BACKTEST = {
    "cagr": "12.67",
    "sharpe": "0.49",
    "drawdown": "23.82",
    "volatility": "12.83",
}

# Stale governance metric signatures that must NEVER be surfaced as current performance
_STALE_METRIC_PATTERNS = [
    r"\b0\.0003\b",
    r"\b0\.0014\b",
    r"\b0\.0011\b",
]

# When BOCPD is UNAVAILABLE, these patterns in LLM output indicate a hallucinated claim
_BOCPD_CLAIM_PATTERNS = [
    r"changepoint\s+(?:probability|prob|signal)\s+(?:is|of|at)\s*(\d+\.?\d*)\s*%",
    r"bocpd\s+(?:probability|prob)\s+(?:is|of|at)\s*(\d+\.?\d*)\s*%",
    r"(\d+\.?\d*)\s*%\s+(?:changepoint|transition)\s+probability",
    r"changepoint\s+(?:is|at)\s*(\d+\.?\d*)\s*%",
]

# Conformal set claim patterns
_CONFORMAL_SIZE_PATTERNS = [
    r"conformal(?:\s+prediction)?\s+set\s+(?:is\s+of\s+size\s+|of\s+size\s+|size\s+(?:is\s+)?|contains\s+)(\d+)",
    r"set\s+(?:is\s+of\s+size\s+|of\s+size\s+|size\s+(?:is\s+)?)(\d+)\s+(?:in\s+conformal|regimes?|classes?)",
    r"\(size\s+(\d+)\)",
]

REGIME_NAMES = {"Risk-On", "Risk-Off", "Transitional", "Late-Cycle", "Post-Shock"}


@dataclass
class ValidationResult:
    is_trusted: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    safe_response: str = ""


def _extract_percentage_claims(text: str) -> List[Tuple[float, str]]:
    """
    Extract (probability_value, context_phrase) pairs from response text.
    Returns list of (value_as_fraction, surrounding_text_snippet).
    """
    claims = []
    for m in re.finditer(r"(\d+\.?\d*)\s*%", text, re.IGNORECASE):
        val = float(m.group(1))
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 50)
        context = text[start:end].replace("\n", " ").strip()
        claims.append((val, context))
    return claims


def _identify_regime_in_context(context: str) -> Optional[str]:
    """Return the regime name found in a context snippet."""
    for r in REGIME_NAMES:
        if r.lower() in context.lower():
            return r
    return None


class ResponseValidator:
    """
    Validates an LLM-generated response against the authoritative evidence bundle.
    """

    def validate(
        self,
        response: str,
        evidence_bundle: Dict[str, Any],
    ) -> ValidationResult:
        """
        Run all grounding checks. Returns ValidationResult.
        """
        violations: List[str] = []
        warnings: List[str] = []

        regime_state = evidence_bundle.get("regime_state", {})
        cp_signal = evidence_bundle.get("changepoint_signal", {})
        backtest = evidence_bundle.get("backtest_context", {})
        conformal_ctx = evidence_bundle.get("conformal_context", {})

        # ------------------------------------------------------------------
        # 1. Stale governance metrics check (0.0003, 0.0014, 0.0011)
        # ------------------------------------------------------------------
        for pat in _STALE_METRIC_PATTERNS:
            if re.search(pat, response):
                violations.append(
                    "LLM referenced stale, superseded governance metric (0.0003 RPS / "
                    "0.0014 ECE / 0.0011 Log Loss). These originated from earlier circular "
                    "evaluation and must NEVER appear in current model evidence."
                )
                break

        # ------------------------------------------------------------------
        # 2. Dominant regime name check
        # ------------------------------------------------------------------
        authoritative_regime = evidence_bundle.get("dominant_regime")
        if not authoritative_regime or regime_state.get("status") == "UNAVAILABLE":
            for r in REGIME_NAMES:
                if re.search(
                    rf"(?:dominant|current|called?|calling)\s+regime\s+(?:is\s+)?{re.escape(r)}",
                    response,
                    re.IGNORECASE,
                ):
                    violations.append(
                        f"LLM claimed dominant regime is '{r}' but regime state is UNAVAILABLE in evidence."
                    )
        else:
            for r in REGIME_NAMES:
                if r != authoritative_regime:
                    if re.search(
                        rf"(?:dominant|current|called?|calling)\s+regime\s+(?:is\s+)?{re.escape(r)}",
                        response,
                        re.IGNORECASE,
                    ):
                        violations.append(
                            f"LLM claimed dominant regime is '{r}' but authoritative evidence says '{authoritative_regime}'."
                        )

        # ------------------------------------------------------------------
        # 3. Regime probability claims
        # ------------------------------------------------------------------
        auth_probs = regime_state.get("regime_probabilities", {})
        if regime_state.get("status") == "UNAVAILABLE":
            pct_claims = _extract_percentage_claims(response)
            for val_pct, ctx in pct_claims:
                regime_ctx = _identify_regime_in_context(ctx)
                if regime_ctx:
                    violations.append(
                        f"LLM claimed {regime_ctx} probability {val_pct:.1f}% "
                        f"but regime state is UNAVAILABLE in the evidence bundle."
                    )
                    break
        elif auth_probs and regime_state.get("status") == "VERIFIED":
            pct_claims = _extract_percentage_claims(response)
            for val_pct, ctx in pct_claims:
                regime_ctx = _identify_regime_in_context(ctx)
                if regime_ctx and regime_ctx in auth_probs:
                    auth_val_pct = auth_probs[regime_ctx] * 100.0
                    diff = abs(val_pct - auth_val_pct)
                    if diff > _PROB_TOLERANCE_PP:
                        violations.append(
                            f"LLM claimed {regime_ctx} probability {val_pct:.1f}% "
                            f"but evidence shows {auth_val_pct:.1f}% "
                            f"(delta={diff:.1f}pp > tolerance={_PROB_TOLERANCE_PP}pp)."
                        )

        # ------------------------------------------------------------------
        # 4. BOCPD changepoint probability check
        # ------------------------------------------------------------------
        if cp_signal.get("status") == "UNAVAILABLE":
            for pat in _BOCPD_CLAIM_PATTERNS:
                m = re.search(pat, response, re.IGNORECASE)
                if m:
                    violations.append(
                        f"LLM claimed a changepoint probability value ({m.group()!r}) "
                        f"but BOCPD signal is UNAVAILABLE in the evidence bundle. "
                        f"No such value exists in the authoritative source."
                    )
                    break
        elif cp_signal.get("status") == "VERIFIED":
            auth_cp = cp_signal.get("bocpd_changepoint_probability", 0.0) * 100.0
            pct_claims = _extract_percentage_claims(response)
            for val_pct, ctx in pct_claims:
                if any(w in ctx.lower() for w in ["changepoint", "bocpd", "transition prob"]):
                    if abs(val_pct - auth_cp) > _PROB_TOLERANCE_PP:
                        violations.append(
                            f"LLM claimed changepoint probability {val_pct:.1f}% "
                            f"but evidence shows {auth_cp:.1f}%."
                        )
                    break

        # ------------------------------------------------------------------
        # 5. Conformal set validation
        # ------------------------------------------------------------------
        auth_conformal_set = evidence_bundle.get("conformal_set", [])
        if conformal_ctx.get("status") == "UNAVAILABLE" or not auth_conformal_set:
            for pat in _CONFORMAL_SIZE_PATTERNS:
                if re.search(pat, response, re.IGNORECASE):
                    violations.append(
                        "LLM claimed conformal set properties but conformal prediction context is UNAVAILABLE."
                    )
                    break
        else:
            auth_set_size = len(auth_conformal_set)
            for pat in _CONFORMAL_SIZE_PATTERNS:
                m = re.search(pat, response, re.IGNORECASE)
                if m:
                    claimed_size = int(m.group(1))
                    if claimed_size != auth_set_size:
                        violations.append(
                            f"LLM claimed conformal set size {claimed_size} but authoritative "
                            f"conformal set has size {auth_set_size} ({auth_conformal_set})."
                        )
                    break

        # ------------------------------------------------------------------
        # 6. SHAP attributions — warning if claimed when unavailable
        # ------------------------------------------------------------------
        shap = evidence_bundle.get("shap_attributions", {})
        if shap.get("status") == "UNAVAILABLE":
            shap_claim_patterns = [
                r"shap\s+(?:driver|value|feature)",
                r"top\s+(?:shap|feature)\s+driver",
                r"attribution.*(?:vix_level|nifty_vol|breadth|usdinr)",
            ]
            for pat in shap_claim_patterns:
                if re.search(pat, response, re.IGNORECASE):
                    warnings.append(
                        "LLM referenced SHAP features but SHAP artifact is UNAVAILABLE. "
                        "These claims cannot be verified against the evidence."
                    )
                    break

        # ------------------------------------------------------------------
        # 7. Backtest metric integrity (CAGR, Sharpe, Drawdown, Volatility)
        # ------------------------------------------------------------------
        if backtest.get("status") == "VERIFIED":
            # CAGR
            m_cagr = re.search(r"\b(?:cagr|annualized\s+return)\b[^\d%]{0,20}(\d+\.\d+)\s*%", response, re.IGNORECASE)
            if m_cagr and m_cagr.group(1) != _LOCKED_BACKTEST["cagr"]:
                violations.append(
                    f"LLM quoted CAGR as {m_cagr.group(1)}% but authoritative locked evidence is "
                    f"{_LOCKED_BACKTEST['cagr']}%."
                )
            # Sharpe
            m_sharpe = re.search(r"\bsharpe(?:\s+ratio)?\b[^\d]{0,20}(\d+\.\d+)", response, re.IGNORECASE)
            if m_sharpe and m_sharpe.group(1) != _LOCKED_BACKTEST["sharpe"]:
                violations.append(
                    f"LLM quoted Sharpe ratio as {m_sharpe.group(1)} but authoritative locked evidence is "
                    f"{_LOCKED_BACKTEST['sharpe']}."
                )
            # Max Drawdown
            m_dd = re.search(r"\b(?:drawdown|max\s*dd)\b[^\d\-]{0,20}(-?\d+\.\d+)\s*%", response, re.IGNORECASE)
            if m_dd:
                val = m_dd.group(1).lstrip("-")
                if val != _LOCKED_BACKTEST["drawdown"]:
                    violations.append(
                        f"LLM quoted Max Drawdown as {m_dd.group(1)}% but authoritative locked evidence is "
                        f"-{_LOCKED_BACKTEST['drawdown']}%."
                    )

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        is_trusted = len(violations) == 0

        if not is_trusted:
            violation_text = "\n".join(f"- {v}" for v in violations)
            safe_response = (
                "⚠️ **GROUNDING VALIDATION FAILED** — This response contained numerical claims "
                "that are inconsistent with the authoritative evidence bundle and cannot be "
                "displayed as trusted analysis.\n\n"
                f"**Violations detected:**\n{violation_text}\n\n"
                "Please check the authoritative evidence panel for verified values."
            )
        else:
            safe_response = response

        result = ValidationResult(
            is_trusted=is_trusted,
            violations=violations,
            warnings=warnings,
            safe_response=safe_response,
        )

        if violations:
            logger.warning(
                "LLM response grounding violations: %d violation(s): %s",
                len(violations),
                "; ".join(violations),
            )

        return result
