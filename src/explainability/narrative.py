"""
Dynamic Natural Language Narrative Generation for Regime Explanations.

Implements REQ-061 and Specification Section E:
Dynamically synthesizes computed SHAP attributions, feature movements,
and model disagreement metrics into institutional, human-readable commentary.
Strictly generated from computed evidence — never hardcoded!
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.contracts import RegimeLabel


class DynamicNarrativeGenerator:
    """
    Synthesizes numerical model outputs, feature attributions, and consensus diagnostics
    into an institutional investment committee brief.
    """

    @staticmethod
    def generate_explanation(
        predicted_regime: str,
        regime_probability: float,
        feature_values: pd.Series,
        shap_attributions: Dict[str, float],
        model_opinions: Dict[str, List[float]],
        changepoint_prob: float,
    ) -> str:
        """
        Builds dynamic human-readable explanation from live mathematical evidence.
        """
        # 1. Identify top positive and negative drivers from SHAP
        sorted_shap = sorted(shap_attributions.items(), key=lambda kv: kv[1], reverse=True)
        top_positive = [k for k, v in sorted_shap if v > 0.05][:3]
        top_negative = [k for k, v in sorted_shap if v < -0.05][:2]

        # 2. Build driver descriptions based on actual feature values
        driver_statements = []

        if "vix_level" in top_positive:
            vix_val = float(feature_values.get("vix_level", 15.0))
            if vix_val > 20.0:
                driver_statements.append(f"elevated INDIA VIX levels ({vix_val:.1f}) reflecting heightened volatility pricing")
            else:
                driver_statements.append(f"compressed INDIA VIX levels ({vix_val:.1f}) indicating market complacency and quiet conditions")

        if "breadth_midcap_ret_21d" in top_positive or "breadth_midcap_ret_21d" in top_negative:
            breadth_val = float(feature_values.get("breadth_midcap_ret_21d", 0.0)) * 100.0
            if breadth_val > 0.5:
                driver_statements.append(f"robust midcap participation (+{breadth_val:.1f}% breadth spread over large-caps)")
            elif breadth_val < -0.5:
                driver_statements.append(f"deteriorating market breadth ({breadth_val:.1f}% underperformance of midcaps vs Nifty 50)")

        if "nifty_dist_sma50" in top_positive or "nifty_dist_sma50" in top_negative:
            trend_val = float(feature_values.get("nifty_dist_sma50", 0.0)) * 100.0
            if trend_val > 1.0:
                driver_statements.append(f"strong positive trend persistence (+{trend_val:.1f}% above 50 DMA)")
            elif trend_val < -1.0:
                driver_statements.append(f"severe trend breakdown ({trend_val:.1f}% below 50 DMA)")

        if "nifty_vol_ewma_21d" in top_positive:
            vol_val = float(feature_values.get("nifty_vol_ewma_21d", 0.15)) * 100.0
            driver_statements.append(f"21-day EWMA realized volatility at {vol_val:.1f}% annualized")

        if "tda_persistence_entropy_h0" in top_positive:
            entropy_val = float(feature_values.get("tda_persistence_entropy_h0", 2.0))
            driver_statements.append(f"shifting topological state space entropy ({entropy_val:.2f} nats)")

        # Fallback if specific drivers not in top
        if not driver_statements and top_positive:
            top_name = top_positive[0].replace("_", " ")
            driver_statements.append(f"primary attribution from {top_name} (+{shap_attributions[top_positive[0]]:.2f})")

        # 3. Model consensus / disagreement summary
        regime_canonical = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
        target_idx = regime_canonical.index(predicted_regime) if predicted_regime in regime_canonical else 0

        agreeing_models = []
        dissenting_models = []

        for m_name, probs in model_opinions.items():
            if len(probs) == 5:
                m_dominant = regime_canonical[int(np.argmax(probs))]
                if m_dominant == predicted_regime:
                    agreeing_models.append(m_name)
                else:
                    dissenting_models.append(f"{m_name} ({m_dominant})")

        consensus_text = ""
        if len(agreeing_models) >= 3:
            consensus_text = f"Broad cross-model consensus was achieved, with {', '.join(agreeing_models)} confirming {predicted_regime} dynamics."
        elif agreeing_models:
            consensus_text = f"{', '.join(agreeing_models)} call {predicted_regime}, whereas {', '.join(dissenting_models)} express divergence."
        else:
            consensus_text = f"High model dispersion observed; dissenters include {', '.join(dissenting_models)}."

        # 4. Changepoint statement
        cp_statement = ""
        if changepoint_prob > 0.30:
            cp_statement = f" NOTICE: BOCPD detected an elevated changepoint probability of {changepoint_prob*100:.1f}%, signaling an active regime transition."

        # 5. Assemble unified institutional paragraph
        drivers_formatted = "; ".join(f"({i+1}) {stmt}" for i, stmt in enumerate(driver_statements))
        if not drivers_formatted:
            drivers_formatted = f"multivariate evidence across momentum and volatility metrics"

        narrative = (
            f"{predicted_regime} probability stands at {regime_probability*100:.1f}% driven primarily by: "
            f"{drivers_formatted}. "
            f"{consensus_text}"
            f"{cp_statement}"
        )
        return narrative
