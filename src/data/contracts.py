"""
Data Contracts and Schema Definitions.

Strict typed representations for financial series, observations, and immutable snapshots.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import json


@dataclass(frozen=True)
class MarketObservation:
    """Represents a single bar or point-in-time observation of a financial instrument."""
    timestamp: datetime
    symbol: str
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    source: str = "NSE"
    quality_flags: int = 0

    def __post_init__(self) -> None:
        if self.close < 0:
            raise ValueError(f"Close price cannot be negative: {self.close}")
        if self.high is not None and self.low is not None:
            if self.high < self.low:
                raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")


@dataclass(frozen=True)
class DataSnapshotMetadata:
    """Metadata describing an immutable point-in-time snapshot of the market database."""
    snapshot_id: str
    creation_timestamp: datetime
    start_date: str
    end_date: str
    assets: List[str]
    sha256_checksum: str
    record_count: int
    missing_data_strategy: str
    description: str = ""

    @classmethod
    def create(
        cls,
        start_date: str,
        end_date: str,
        assets: List[str],
        content_bytes: bytes,
        record_count: int,
        missing_strategy: str,
        description: str = "",
    ) -> "DataSnapshotMetadata":
        checksum = hashlib.sha256(content_bytes).hexdigest()
        snap_id = f"data_snap_{start_date[:4]}_{end_date[:4]}_{checksum[:8]}"
        return cls(
            snapshot_id=snap_id,
            creation_timestamp=datetime.utcnow(),
            start_date=start_date,
            end_date=end_date,
            assets=sorted(assets),
            sha256_checksum=checksum,
            record_count=record_count,
            missing_data_strategy=missing_strategy,
            description=description,
        )


@dataclass(frozen=True)
class FeatureSnapshotMetadata:
    """Metadata describing an engineered feature matrix."""
    feature_snapshot_id: str
    data_snapshot_id: str
    creation_timestamp: datetime
    feature_names: List[str]
    start_date: str
    end_date: str
    lookback_buffer_days: int
    sha256_checksum: str
    normalization_policy: str
