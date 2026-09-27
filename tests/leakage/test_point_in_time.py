"""
Point-in-Time Availability & Future Publication Leakage Tests.

Adversarially proves that future publications cannot contaminate historical predictions.
"""

from datetime import date, datetime
import pandas as pd
import pytest

from src.data.contracts import (
    AvailabilityStatus,
    Frequency,
    LagMethodology,
    PointInTimeRecord,
)
from src.data.point_in_time import PointInTimeGuard


@pytest.mark.leakage
def test_point_in_time_availability_filtering() -> None:
    """
    Test that observations are only visible on or after their effective_availability_date.
    Example: January observation published on February 10th.
    """
    rec_jan = PointInTimeRecord(
        series_id="AMFI_SIP",
        observation_date=date(2023, 1, 31),
        effective_availability_date=date(2023, 2, 10),
        source="AMFI",
        frequency=Frequency.MONTHLY,
        units="inr_crores",
        raw_value=13856.0,
        cleaned_value=13856.0,
        lag_methodology=LagMethodology.OFFICIAL_CALENDAR,
        availability_status=AvailabilityStatus.VERIFIED,
        publication_date=date(2023, 2, 10),
        release_lag_days=10,
    )

    # Prediction as of Jan 31: record MUST NOT be available
    as_of_jan31 = datetime(2023, 1, 31, 15, 30)
    assert rec_jan.is_available_at(as_of_jan31) is False
    filtered_jan = PointInTimeGuard.filter_by_availability([rec_jan], as_of_jan31)
    assert len(filtered_jan) == 0

    # Prediction as of Feb 9: record MUST NOT be available
    as_of_feb9 = datetime(2023, 2, 9, 15, 30)
    assert rec_jan.is_available_at(as_of_feb9) is False

    # Prediction as of Feb 10: record IS now available
    as_of_feb10 = datetime(2023, 2, 10, 15, 30)
    assert rec_jan.is_available_at(as_of_feb10) is True
    filtered_feb = PointInTimeGuard.filter_by_availability([rec_jan], as_of_feb10)
    assert len(filtered_feb) == 1


@pytest.mark.leakage
def test_adversarial_future_publication_leakage_assertion() -> None:
    """
    Adversarial Leakage Test:
    Ensures that a future publication cannot enter a feature row at as_of_date.
    """
    dates = pd.bdate_range("2023-01-01", "2023-03-31")
    raw_prices = pd.DataFrame({"close": range(len(dates))}, index=dates)

    as_of_date = pd.Timestamp("2023-02-15")

    # Align series with a 1-day reporting lag (e.g. FII flow T+1)
    lagged_aligned = PointInTimeGuard.align_series_to_point_in_time(
        series_df=raw_prices,
        availability_lag_days=1,
        as_of_date=as_of_date,
    )

    # Must not contain observation from 2023-02-15 because with lag_days=1, it is available on 2023-02-16!
    assert pd.Timestamp("2023-02-15") not in lagged_aligned.index
    # Latest observation must be 2023-02-14 or earlier
    assert lagged_aligned.index.max() <= pd.Timestamp("2023-02-14")

    # Assertion helper must pass
    PointInTimeGuard.assert_no_future_leakage(
        feature_matrix=lagged_aligned,
        raw_series_df=raw_prices,
        as_of_date=as_of_date,
        availability_lag_days=1,
    )

    # Corrupt feature matrix with future date: assertion must raise AssertionError
    corrupted_matrix = raw_prices.loc[raw_prices.index <= pd.Timestamp("2023-02-20")]
    with pytest.raises(AssertionError, match="Leakage violation"):
        PointInTimeGuard.assert_no_future_leakage(
            feature_matrix=corrupted_matrix,
            raw_series_df=raw_prices,
            as_of_date=as_of_date,
            availability_lag_days=1,
        )
