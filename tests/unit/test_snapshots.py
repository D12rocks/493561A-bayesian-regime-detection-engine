"""
Unit tests for immutable dataset snapshot creation and cryptographic verification.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.contracts import SeriesMetadata
from src.data.snapshot import DataSnapshotEngine


@pytest.mark.unit
def test_snapshot_creation_and_integrity(tmp_path: Path) -> None:
    engine = DataSnapshotEngine(data_root=tmp_path)

    dates = pd.bdate_range("2020-01-01", periods=10)
    df1 = pd.DataFrame({"close": np.linspace(100, 110, 10)}, index=dates)
    df2 = pd.DataFrame({"close": np.linspace(20, 15, 10)}, index=dates)

    meta1 = SeriesMetadata(
        series_id="ASSET_A",
        series_name="Asset A",
        source="Test",
        retrieval_timestamp=datetime.utcnow(),
        start_date="2020-01-01",
        end_date="2020-01-14",
        record_count=10,
        frequency="daily",
        units="points",
    )
    meta2 = SeriesMetadata(
        series_id="ASSET_B",
        series_name="Asset B",
        source="Test",
        retrieval_timestamp=datetime.utcnow(),
        start_date="2020-01-01",
        end_date="2020-01-14",
        record_count=10,
        frequency="daily",
        units="points",
    )

    snapshot_meta = engine.create_snapshot(
        series_dict={"ASSET_A": df1, "ASSET_B": df2},
        metadata_dict={"ASSET_A": meta1, "ASSET_B": meta2},
        start_date="2020-01-01",
        end_date="2020-01-14",
        snapshot_tag="test_snap",
    )

    assert snapshot_meta.total_records == 20
    assert len(snapshot_meta.sha256_checksum) == 64  # Valid SHA-256 string
    assert engine.verify_snapshot_integrity(snapshot_meta.snapshot_id) is True

    # Test loading
    loaded_data, loaded_meta = engine.load_snapshot(snapshot_meta.snapshot_id)
    assert "ASSET_A" in loaded_data
    assert "ASSET_B" in loaded_data
    assert len(loaded_data["ASSET_A"]) == 10
    pd.testing.assert_series_equal(loaded_data["ASSET_A"]["close"], df1["close"], check_freq=False)


@pytest.mark.unit
def test_snapshot_tamper_detection(tmp_path: Path) -> None:
    engine = DataSnapshotEngine(data_root=tmp_path)
    dates = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)
    meta = SeriesMetadata(
        series_id="ASSET_T",
        series_name="Asset T",
        source="Test",
        retrieval_timestamp=datetime.utcnow(),
        start_date="2020-01-01",
        end_date="2020-01-07",
        record_count=5,
        frequency="daily",
        units="points",
    )

    snapshot_meta = engine.create_snapshot(
        series_dict={"ASSET_T": df},
        metadata_dict={"ASSET_T": meta},
        start_date="2020-01-01",
        end_date="2020-01-07",
        snapshot_tag="tamper_test",
    )
    assert engine.verify_snapshot_integrity(snapshot_meta.snapshot_id) is True

    # Tamper with the underlying file
    snap_dir = tmp_path / "snapshots" / snapshot_meta.snapshot_id
    file_to_corrupt = list(snap_dir.glob("ASSET_T.*"))[0]
    with open(file_to_corrupt, "a") as f:
        f.write("\n999999,999.0\n")

    # Integrity verification must now detect corruption
    assert engine.verify_snapshot_integrity(snapshot_meta.snapshot_id) is False
