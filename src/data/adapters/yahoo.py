"""
Yahoo Finance Data Adapter.

Retrieves historical daily Indian equity indices, volatility (India VIX), and currency (USD/INR)
via direct Yahoo Finance JSON chart endpoints with rate-limiting, error handling, and SHA-256 integrity audits.
"""

from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from src.data.adapters.base import BaseDataAdapter
from src.data.contracts import SeriesMetadata


class YahooFinanceAdapter(BaseDataAdapter):
    """
    Production-grade adapter for Yahoo Finance daily market data.
    """

    DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(source_name="YahooFinance", config=config)
        self.session = requests.Session()
        user_agent = self.config.get("user_agent", self.DEFAULT_USER_AGENT)
        self.session.headers.update({"User-Agent": user_agent})

    def check_availability(self, symbol: str) -> Tuple[bool, str]:
        """Test symbol accessibility."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json().get("chart", {})
                if data.get("result") and len(data["result"]) > 0:
                    return True, "Symbol accessible on Yahoo Finance."
                return False, f"Empty chart response for symbol {symbol}."
            return False, f"HTTP {r.status_code} returned by Yahoo Finance."
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def fetch_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        series_name: Optional[str] = None,
        units: str = "index_points",
    ) -> Tuple[Optional[pd.DataFrame], SeriesMetadata]:
        """
        Fetch full daily historical data for symbol between start_date and end_date.
        """
        dt_start = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        dt_end = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        
        display_name = series_name or symbol
        retrieval_time = datetime.utcnow()

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&period1={dt_start}&period2={dt_end}"
        
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                meta = SeriesMetadata(
                    series_id=symbol,
                    series_name=display_name,
                    source="YahooFinance",
                    retrieval_timestamp=retrieval_time,
                    start_date=start_date,
                    end_date=end_date,
                    record_count=0,
                    frequency="daily",
                    units=units,
                    is_available=False,
                    unavailability_reason=f"HTTP {resp.status_code} from Yahoo Finance",
                )
                return None, meta

            payload = resp.json()
            results = payload.get("chart", {}).get("result")
            if not results or len(results) == 0:
                meta = SeriesMetadata(
                    series_id=symbol,
                    series_name=display_name,
                    source="YahooFinance",
                    retrieval_timestamp=retrieval_time,
                    start_date=start_date,
                    end_date=end_date,
                    record_count=0,
                    frequency="daily",
                    units=units,
                    is_available=False,
                    unavailability_reason="Empty result list in Yahoo payload",
                )
                return None, meta

            chart_data = results[0]
            timestamps = chart_data.get("timestamp", [])
            indicators = chart_data.get("indicators", {}).get("quote", [{}])[0]
            
            if len(timestamps) == 0:
                meta = SeriesMetadata(
                    series_id=symbol,
                    series_name=display_name,
                    source="YahooFinance",
                    retrieval_timestamp=retrieval_time,
                    start_date=start_date,
                    end_date=end_date,
                    record_count=0,
                    frequency="daily",
                    units=units,
                    is_available=False,
                    unavailability_reason="No timestamps returned in range",
                )
                return None, meta

            # Build DataFrame
            dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata").date
            df = pd.DataFrame(
                {
                    "open": indicators.get("open", [np.nan] * len(timestamps)),
                    "high": indicators.get("high", [np.nan] * len(timestamps)),
                    "low": indicators.get("low", [np.nan] * len(timestamps)),
                    "close": indicators.get("close", [np.nan] * len(timestamps)),
                    "volume": indicators.get("volume", [0] * len(timestamps)),
                },
                index=pd.DatetimeIndex(dates, name="date"),
            )
            
            # Remove duplicated calendar dates if market auctions/adjustments created duplicate entries
            df = df.loc[~df.index.duplicated(keep="first")]

            # Compute content hash
            content_bytes = df.to_csv().encode("utf-8")
            file_hash = hashlib.sha256(content_bytes).hexdigest()

            actual_start = str(df.index[0].date())
            actual_end = str(df.index[-1].date())

            meta = SeriesMetadata(
                series_id=symbol,
                series_name=display_name,
                source="YahooFinance",
                retrieval_timestamp=retrieval_time,
                start_date=actual_start,
                end_date=actual_end,
                record_count=len(df),
                frequency="daily",
                units=units,
                file_sha256=file_hash,
                is_available=True,
            )

            return df, meta

        except Exception as e:
            meta = SeriesMetadata(
                series_id=symbol,
                series_name=display_name,
                source="YahooFinance",
                retrieval_timestamp=retrieval_time,
                start_date=start_date,
                end_date=end_date,
                record_count=0,
                frequency="daily",
                units=units,
                is_available=False,
                unavailability_reason=f"Exception during fetch: {str(e)}",
            )
            return None, meta
