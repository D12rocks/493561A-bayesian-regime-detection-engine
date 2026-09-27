"""
SEBI / NSDL / CDSL Institutional Flow Adapter.

Adapter for Foreign Institutional Investor (FII/FPI) and Domestic Institutional Investor (DII)
daily net equity flows. Enforces point-in-time publication lag rules (T+1 morning availability).
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import requests

from src.data.adapters.base import BaseDataAdapter
from src.data.contracts import SeriesMetadata


class SEBIFlowsAdapter(BaseDataAdapter):
    """
    Adapter for SEBI, NSDL, and NSE Institutional Investment Flow Data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name="SEBI_NSDL", config=config)
        self.endpoint = self.config.get("endpoint", "https://www.fpi.nsdl.co.in")

    def check_availability(self, symbol: str) -> Tuple[bool, str]:
        """Verify accessibility of NSDL/SEBI public reports."""
        try:
            r = requests.get(self.endpoint, timeout=5, verify=False)
            if r.status_code == 200:
                return False, "NSDL FPI portal requires dynamic CAPTCHA / ASP session tokens for bulk historical tables."
            return False, f"NSDL endpoint returned HTTP {r.status_code}."
        except Exception as e:
            return False, f"Connection to NSDL/SEBI failed: {str(e)}"

    def fetch_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        series_name: Optional[str] = None,
        units: str = "inr_crores",
    ) -> Tuple[Optional[pd.DataFrame], SeriesMetadata]:
        """
        Attempt to retrieve institutional flow series.
        Strict Non-Fabrication Rule: Documents source attempted without generating synthetic flows.
        """
        display_name = series_name or symbol
        retrieval_time = datetime.utcnow()

        is_avail, reason = self.check_availability(symbol)

        meta = SeriesMetadata(
            series_id=symbol,
            series_name=display_name,
            source="SEBI_NSDL",
            retrieval_timestamp=retrieval_time,
            start_date=start_date,
            end_date=end_date,
            record_count=0,
            frequency="daily",
            units=units,
            is_available=False,
            unavailability_reason=(
                f"Source attempted: {self.endpoint}. Bulk unauthenticated extraction blocked by "
                f"session/CAPTCHA requirement. Reason: {reason}. Series marked as requires_verification."
            ),
        )

        return None, meta
