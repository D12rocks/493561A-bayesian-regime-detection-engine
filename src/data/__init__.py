"""Data ingestion, contracts, validation, snapshotting, and quality audit module."""

from src.data.adapters import (
    AMFIAdapter,
    BaseDataAdapter,
    RBIAdapter,
    SEBIFlowsAdapter,
    YahooFinanceAdapter,
)
from src.data.contracts import (
    AvailabilityStatus,
    DataQualityReport,
    DataSnapshotMetadata,
    Frequency,
    LagMethodology,
    MarketObservation,
    PointInTimeRecord,
    SeriesMetadata,
)
from src.data.point_in_time import PointInTimeGuard
from src.data.quality import DataQualityAuditor
from src.data.snapshot import DataSnapshotEngine

__all__ = [
    "MarketObservation",
    "PointInTimeRecord",
    "SeriesMetadata",
    "DataQualityReport",
    "DataSnapshotMetadata",
    "Frequency",
    "LagMethodology",
    "AvailabilityStatus",
    "BaseDataAdapter",
    "YahooFinanceAdapter",
    "RBIAdapter",
    "SEBIFlowsAdapter",
    "AMFIAdapter",
    "DataQualityAuditor",
    "DataSnapshotEngine",
    "PointInTimeGuard",
]
