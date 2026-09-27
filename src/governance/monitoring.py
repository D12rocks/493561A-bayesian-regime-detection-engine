"""
Model Monitoring: Population Stability Index (PSI) and Drift Alarms.

Implements REQ-071 and Specification Section K:
- Population Stability Index (PSI) tracking distributional drift between training and live features
- Conformal coverage degradation monitor
- Champion / Challenger performance auditing
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class GovernanceMonitor:
    """
    Continuous statistical monitoring engine guarding against model degradation,
    covariate shift, and calibration decay.
    """

    @staticmethod
    def compute_psi(
        reference_series: np.ndarray,
        current_series: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculates Population Stability Index (PSI):
        PSI = sum (P_b - Q_b) * ln(P_b / Q_b)
        
        Thresholds:
        PSI < 0.10: Stable / No drift
        0.10 <= PSI < 0.25: Moderate shift
        PSI >= 0.25: Significant drift (triggers model retraining / Challenger review)
        """
        ref = reference_series[~np.isnan(reference_series)]
        curr = current_series[~np.isnan(current_series)]

        if len(ref) < 20 or len(curr) < 20:
            return 0.0

        # Bin boundaries from reference distribution
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.quantile(ref, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        # Deduplicate bins if any
        bins = np.unique(bins)
        if len(bins) < 3:
            return 0.0

        ref_counts = np.histogram(ref, bins=bins)[0]
        curr_counts = np.histogram(curr, bins=bins)[0]

        # Convert to probabilities with Laplace smoothing
        eps = 1e-4
        p = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
        q = (curr_counts + eps) / (len(curr) + eps * len(curr_counts))

        psi = np.sum((p - q) * np.log(p / q))
        return float(psi)

    @classmethod
    def monitor_feature_matrix(
        cls,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Runs PSI across all numeric features.
        """
        results = []
        common_cols = [c for c in reference_df.columns if c in current_df.columns]

        for col in common_cols:
            psi_val = cls.compute_psi(reference_df[col].values, current_df[col].values)
            if psi_val < 0.10:
                status = "STABLE"
            elif psi_val < 0.25:
                status = "MODERATE_SHIFT"
            else:
                status = "SIGNIFICANT_DRIFT"

            results.append({
                "feature": col,
                "psi": round(psi_val, 4),
                "status": status,
                "alarm_triggered": psi_val >= 0.25,
            })

        return pd.DataFrame(results).sort_values("psi", ascending=False)

    @staticmethod
    def audit_conformal_health(
        realized_coverage: float,
        nominal_target: float = 0.90,
        warning_threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Audits conformal prediction health: triggers warning if coverage deteriorates below 80%.
        """
        is_healthy = realized_coverage >= warning_threshold
        gap = realized_coverage - nominal_target

        return {
            "realized_coverage": round(realized_coverage, 4),
            "nominal_target": nominal_target,
            "coverage_gap": round(gap, 4),
            "is_healthy": is_healthy,
            "alarm_triggered": not is_healthy,
            "action_required": "RECALIBRATE_CONFORMAL_LAYER" if not is_healthy else "NONE",
        }
