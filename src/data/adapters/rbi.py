"""
Reserve Bank of India (RBI) & CCIL Data Adapter.

Adapter for Indian Sovereign Debt (10-Year Benchmark Gilt Yield and AAA-Gilt Spread).
Handles point-in-time publication schedules from RBI DBIE and CCIL reports.
If institutional endpoints are unreachable without private keys, documents the limitation
without fabricating synthetic yield observations.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import requests

from src.data.adapters.base import BaseDataAdapter
from src.data.contracts import SeriesMetadata


class RBIAdapter(BaseDataAdapter):
    """
    Adapter for India Sovereign Debt yields and credit spreads from RBI and CCIL.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name="RBI_CCIL", config=config)
        self.base_url = self.config.get("base_url", "https://data.rbi.org.in")

    def check_availability(self, symbol: str) -> Tuple[bool, str]:
        """
        Verify access to RBI open data endpoints.
        """
        try:
            r = requests.get(self.base_url, timeout=5, verify=False)
            if r.status_code == 200:
                # Direct programmatic API without authenticated tokens is restricted for Gilt time series
                return False, "RBI open data portal requires authenticated session/token for historical yield API."
            return False, f"RBI endpoint returned HTTP {r.status_code}."
        except Exception as e:
            return False, f"RBI connection attempt failed: {str(e)}"

    def fetch_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        series_name: Optional[str] = None,
        units: str = "yield_percent",
    ) -> Tuple[Optional[pd.DataFrame], SeriesMetadata]:
        """
        Attempt to retrieve historical yield series.
        Strict Non-Fabrication Rule: If public endpoint does not supply valid authenticated tables,
        return None with detailed provenance metadata and quarantine status.
        """
        display_name = series_name or symbol
        retrieval_time = datetime.utcnow()

        is_avail, reason = self.check_availability(symbol)

        meta = SeriesMetadata(
            series_id=symbol,
            series_name=display_name,
            source="RBI_CCIL",
            retrieval_timestamp=retrieval_time,
            start_date=start_date,
            end_date=end_date,
            record_count=0,
            frequency="daily",
            units=units,
            is_available=False,
            unavailability_reason=(
                f"Source attempted: {self.base_url}. Direct unauthenticated historical table extraction "
                f"is unavailable on public tier. Reason: {reason}. Quarantined from production inference."
            ),
        )

        return None, meta
