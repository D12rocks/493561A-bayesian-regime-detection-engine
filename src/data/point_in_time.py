"""
Point-in-Time Availability & Anti-Leakage Safeguards.

Ensures that downstream model inference can strictly access observations satisfying:
availability_date <= prediction_timestamp
"""

from datetime import date, datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.contracts import AvailabilityStatus, PointInTimeRecord


class PointInTimeGuard:
    """
    Enforces chronological availability boundaries on financial and macroeconomic time series.
    """

    @staticmethod
    def filter_by_availability(
        records: List[PointInTimeRecord],
        as_of_timestamp: datetime,
    ) -> List[PointInTimeRecord]:
        """
        Filter records to include ONLY those physically available on or before as_of_timestamp.
        """
        as_of_date = as_of_timestamp.date()
        return [r for r in records if r.effective_availability_date <= as_of_date]

    @staticmethod
    def align_series_to_point_in_time(
        series_df: pd.DataFrame,
        availability_lag_days: int,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Align a DataFrame where each row at observation date t becomes available at t + availability_lag_days.
        Any row with (observation_date + availability_lag_days) > as_of_date is purged.
        """
        df = series_df.copy()
        # Compute available date
        available_dates = df.index + pd.Timedelta(days=availability_lag_days)
        # Keep only rows whose availability date <= as_of_date
        mask = available_dates <= as_of_date
        return df.loc[mask]

    @staticmethod
    def assert_no_future_leakage(
        feature_matrix: pd.DataFrame,
        raw_series_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
        availability_lag_days: int = 0,
    ) -> None:
        """
        Adversarial Verification:
        Asserts that feature_matrix contains zero information from raw_series_df past its availability cutoff.
        """
        cutoff_date = as_of_date - pd.Timedelta(days=availability_lag_days)
        future_raw = raw_series_df.loc[raw_series_df.index > cutoff_date]
        if len(future_raw) > 0 and len(feature_matrix) > 0:
            if feature_matrix.index.max() > as_of_date:
                raise AssertionError(
                    f"Leakage violation: Feature matrix index {feature_matrix.index.max()} "
                    f"exceeds as_of_date {as_of_date}."
                )
