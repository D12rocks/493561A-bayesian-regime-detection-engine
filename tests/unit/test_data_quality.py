"""
Unit tests for data quality audit engine.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.data.quality import DataQualityAuditor


@pytest.mark.unit
def test_auditor_clean_series() -> None:
    auditor = DataQualityAuditor()
    dates = pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-02"])
    df = pd.DataFrame({"close": [102.0, 100.0, 102.5]}, index=dates)

    cleaned = auditor.clean_series(df)
    assert len(cleaned) == 2
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates


@pytest.mark.unit
def test_auditor_detects_duplicates() -> None:
    auditor = DataQualityAuditor()
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"])
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)

    report = auditor.audit_series(df, series_id="TEST_SERIES")
    assert report.duplicate_count == 1
    assert report.is_date_strictly_monotonic is False


@pytest.mark.unit
def test_auditor_detects_missing_values() -> None:
    auditor = DataQualityAuditor()
    dates = pd.bdate_range("2020-01-01", periods=10)
    values = [100.0, np.nan, 102.0, np.nan, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    df = pd.DataFrame({"close": values}, index=dates)

    report = auditor.audit_series(df, series_id="TEST_SERIES")
    assert report.missing_count == 2
    assert report.missing_percentage == 20.0


@pytest.mark.unit
def test_auditor_detects_impossible_negative_price() -> None:
    auditor = DataQualityAuditor()
    dates = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"close": [100.0, 101.0, -50.0, 102.0, 103.0]}, index=dates)

    report = auditor.audit_series(df, series_id="TEST_EQUITY")
    assert report.impossible_values_count >= 1
    assert report.status == "FAILED"


@pytest.mark.unit
def test_auditor_detects_stale_prices() -> None:
    auditor = DataQualityAuditor(max_acceptable_stale_days=3)
    dates = pd.bdate_range("2020-01-01", periods=6)
    # Stale across 4 consecutive days
    df = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0, 101.0, 102.0]}, index=dates)

    report = auditor.audit_series(df, series_id="TEST_STALE")
    assert report.stale_values_count >= 1
