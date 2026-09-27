"""
Unit tests for data adapters and schema normalization.
"""

from datetime import datetime
import pandas as pd
import pytest

from src.data.adapters import (
    AMFIAdapter,
    RBIAdapter,
    SEBIFlowsAdapter,
    YahooFinanceAdapter,
)


@pytest.mark.unit
def test_yahoo_adapter_metadata_structure() -> None:
    adapter = YahooFinanceAdapter()
    assert adapter.source_name == "YahooFinance"
    is_avail, msg = adapter.check_availability("^NSEI")
    assert isinstance(is_avail, bool)
    assert isinstance(msg, str)


@pytest.mark.unit
def test_rbi_adapter_quarantine_integrity() -> None:
    adapter = RBIAdapter()
    assert adapter.source_name == "RBI_CCIL"
    df, meta = adapter.fetch_series("GILT_10Y", "2020-01-01", "2020-12-31")
    assert df is None  # Strict non-fabrication
    assert meta.is_available is False
    assert "Direct unauthenticated historical table extraction is unavailable" in meta.unavailability_reason


@pytest.mark.unit
def test_sebi_adapter_quarantine_integrity() -> None:
    adapter = SEBIFlowsAdapter()
    assert adapter.source_name == "SEBI_NSDL"
    df, meta = adapter.fetch_series("FII_NET_EQUITY", "2020-01-01", "2020-12-31")
    assert df is None  # Strict non-fabrication
    assert meta.is_available is False
    assert "requires_verification" in meta.unavailability_reason


@pytest.mark.unit
def test_amfi_adapter_quarantine_integrity() -> None:
    adapter = AMFIAdapter()
    assert adapter.source_name == "AMFI"
    df, meta = adapter.fetch_series("SIP_TOTAL", "2020-01-01", "2020-12-31")
    assert df is None  # Strict non-fabrication
    assert meta.is_available is False
    assert "press release audit" in meta.unavailability_reason
