"""
Feature Stationarity and Collinearity Diagnostic Suite.

Implements statistical stationarity verification (ADF test) and
Variance Inflation Factor (VIF) collinearity screening.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureAuditor:
    """
    Evaluates feature matrices for time-series stationarity and multicollinearity.
    """

    @staticmethod
    def test_stationarity(
        df_features: pd.DataFrame, significance_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Runs the Augmented Dickey-Fuller (ADF) test on each non-constant column.
        
        Returns:
            DataFrame with columns: ['feature', 'adf_stat', 'p_value', 'crit_5pct', 'is_stationary']
        """
        results = []
        for col in df_features.columns:
            series = df_features[col].dropna()
            if len(series) < 30 or series.nunique() <= 1:
                results.append({
                    "feature": col,
                    "adf_stat": np.nan,
                    "p_value": 1.0,
                    "crit_5pct": np.nan,
                    "is_stationary": False,
                    "status": "INSUFFICIENT_VARIANCE",
                })
                continue
                
            try:
                adf_res = adfuller(series.values, autolag="AIC")
                stat = float(adf_res[0])
                pval = float(adf_res[1])
                crit_5 = float(adf_res[4]["5%"])
                is_stat = pval < significance_level
                status = "STATIONARY" if is_stat else "NON_STATIONARY"
                
                results.append({
                    "feature": col,
                    "adf_stat": round(stat, 4),
                    "p_value": round(pval, 6),
                    "crit_5pct": round(crit_5, 4),
                    "is_stationary": is_stat,
                    "status": status,
                })
            except Exception as e:
                results.append({
                    "feature": col,
                    "adf_stat": np.nan,
                    "p_value": 1.0,
                    "crit_5pct": np.nan,
                    "is_stationary": False,
                    "status": f"ERROR: {str(e)[:30]}",
                })
                
        return pd.DataFrame(results)

    @staticmethod
    def compute_vif(df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Variance Inflation Factor (VIF) for numeric feature matrix.
        VIF > 10 indicates high multicollinearity.
        """
        clean_df = df_features.dropna().select_dtypes(include=[np.number])
        # Drop constant columns
        clean_df = clean_df.loc[:, clean_df.nunique() > 1]
        
        # Standardize for numerical stability
        standardized = (clean_df - clean_df.mean()) / (clean_df.std() + 1e-8)
        
        vif_data = []
        x_mat = standardized.values
        cols = clean_df.columns
        
        for i, col in enumerate(cols):
            try:
                vif = variance_inflation_factor(x_mat, i)
                vif_data.append({
                    "feature": col,
                    "vif": round(float(vif), 2),
                    "collinear_warning": vif > 10.0,
                })
            except Exception:
                vif_data.append({
                    "feature": col,
                    "vif": np.nan,
                    "collinear_warning": False,
                })
                
        return pd.DataFrame(vif_data).sort_values("vif", ascending=False)
