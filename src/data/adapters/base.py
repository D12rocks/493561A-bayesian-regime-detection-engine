"""
Base Data Source Adapter Interface.

Defines the contract for fetching, normalizing, and auditing external market data sources.
Every adapter must normalize raw data into standard time-series DataFrames with full provenance metadata.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.data.contracts import SeriesMetadata


class BaseDataAdapter(ABC):
    """
    Abstract interface for all source-specific data ingestion adapters.
    """

    def __init__(self, source_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.source_name = source_name
        self.config = config or {}

    @abstractmethod
    def fetch_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[Optional[pd.DataFrame], SeriesMetadata]:
        """
        Fetch raw data for symbol across [start_date, end_date].
        Returns:
            Tuple[Optional[pd.DataFrame], SeriesMetadata]:
                - DataFrame indexed by DatetimeIndex (Asia/Kolkata), columns normalized.
                - Metadata capturing retrieval timestamp, record count, SHA-256 hash, and availability status.
                If source is inaccessible, returns (None, metadata with is_available=False and reason).
        """
        pass

    @abstractmethod
    def check_availability(self, symbol: str) -> Tuple[bool, str]:
        """Verify whether the data source and symbol are accessible."""
        pass
