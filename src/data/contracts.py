"""
Data Contracts and Schema Definitions.

Strict typed representations for financial series, observations, point-in-time metadata,
quality audits, and immutable dataset snapshots.
"""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional
import numpy as np


class Frequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class LagMethodology(str, Enum):
    SAME_DAY_EOD = "same_day_eod"
    T_PLUS_1 = "t_plus_1"
    OFFICIAL_CALENDAR = "official_calendar"
    ESTIMATED_CALENDAR = "estimated_calendar"
    UNVERIFIED_PENDING_AUDIT = "unverified_pending_audit"


class AvailabilityStatus(str, Enum):
    VERIFIED = "verified"
    REQUIRES_VERIFICATION = "requires_verification"
    UNAVAILABLE_SOURCE_ATTEMPTED = "unavailable_source_attempted"


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
class PointInTimeRecord:
    """
    Enforces point-in-time safety by tracking when an observation was physically available to trade.
    """
    series_id: str
    observation_date: date
    effective_availability_date: date
    source: str
    frequency: Frequency
    units: str
    raw_value: float
    cleaned_value: float
    lag_methodology: LagMethodology
    availability_status: AvailabilityStatus
    publication_date: Optional[date] = None
    release_lag_days: int = 0
    transformation_metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available_at(self, prediction_timestamp: datetime) -> bool:
        """Point-in-time rule: effective_availability_date <= prediction_timestamp.date()"""
        return self.effective_availability_date <= prediction_timestamp.date()


@dataclass
class SeriesMetadata:
    """Provenance and schema metadata for an ingested financial time series."""
    series_id: str
    series_name: str
    source: str
    retrieval_timestamp: datetime
    start_date: str
    end_date: str
    record_count: int
    frequency: str
    units: str
    schema_version: str = "1.0.0"
    processing_version: str = "1.0.0"
    file_sha256: str = ""
    is_available: bool = True
    unavailability_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["retrieval_timestamp"] = self.retrieval_timestamp.isoformat()
        return d


@dataclass
class DataQualityReport:
    """Comprehensive data quality audit outcome."""
    series_id: str
    total_records: int
    duplicate_count: int
    missing_count: int
    missing_percentage: float
    is_date_strictly_monotonic: bool
    unexpected_gaps_count: int
    impossible_values_count: int
    stale_values_count: int
    timezone_normalized: bool
    missing_value_action: str
    rationale: str
    status: str  # "PASSED", "WARNING", "FAILED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataSnapshotMetadata:
    """Metadata describing an immutable point-in-time snapshot of the market database."""
    snapshot_id: str
    creation_timestamp: datetime
    start_date: str
    end_date: str
    series_manifest: Dict[str, SeriesMetadata]
    sha256_checksum: str
    total_records: int
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "sha256_checksum": self.sha256_checksum,
            "total_records": self.total_records,
            "schema_version": self.schema_version,
            "series": {k: v.to_dict() for k, v in self.series_manifest.items()},
        }
