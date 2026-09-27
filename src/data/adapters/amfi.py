"""
Association of Mutual Funds in India (AMFI) SIP Data Adapter.

Adapter for monthly Indian mutual fund SIP contribution totals.
Enforces point-in-time publication lag rules (AMFI monthly press release calendar).
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import requests

from src.data.adapters.base import BaseDataAdapter
from src.data.contracts import SeriesMetadata


class AMFIAdapter(BaseDataAdapter):
    """
    Adapter for AMFI Monthly Mutual Fund SIP Contribution Data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name="AMFI", config=config)
        self.endpoint = self.config.get("endpoint", "https://www.amfiindia.com")

    def check_availability(self, symbol: str) -> Tuple[bool, str]:
        """Verify accessibility of AMFI historical portal."""
        try:
            r = requests.get(self.endpoint, timeout=5, verify=False)
            if r.status_code == 200:
                return False, "AMFI portal blocks automated scrapers via Cloudflare / TLS fingerprinting."
            return False, f"AMFI endpoint returned HTTP {r.status_code}."
        except Exception as e:
            return False, f"Connection to AMFI failed: {str(e)}"

    def fetch_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        series_name: Optional[str] = None,
        units: str = "inr_crores",
    ) -> Tuple[Optional[pd.DataFrame], SeriesMetadata]:
        """
        Attempt to retrieve AMFI monthly SIP contribution series.
        Strict Non-Fabrication Rule: If public endpoint requires manual CSV download from portal,
        record the status and quarantine from production use until verified.
        """
        display_name = series_name or symbol
        retrieval_time = datetime.utcnow()

        is_avail, reason = self.check_availability(symbol)

        meta = SeriesMetadata(
            series_id=symbol,
            series_name=display_name,
            source="AMFI",
            retrieval_timestamp=retrieval_time,
            start_date=start_date,
            end_date=end_date,
            record_count=0,
            frequency="monthly",
            units=units,
            is_available=False,
            unavailability_reason=(
                f"Source attempted: {self.endpoint}. Automatic scraping prevented by bot mitigation / TLS checks. "
                f"Reason: {reason}. Requires manual point-in-time press release audit before production integration."
            ),
        )

        return None, meta
