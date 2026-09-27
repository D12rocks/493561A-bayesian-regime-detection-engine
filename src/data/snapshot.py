"""
Immutable Data Snapshot Engine.

Serializes ingested and cleaned market datasets into immutable point-in-time snapshots
with cryptographic SHA-256 integrity verification and manifests.
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.data.contracts import DataSnapshotMetadata, SeriesMetadata
from src.utils.config import get_project_root


class DataSnapshotEngine:
    """
    Manages immutable point-in-time dataset snapshots with SHA-256 cryptographic verification.
    """

    def __init__(self, data_root: Optional[Path] = None) -> None:
        if data_root is None:
            data_root = get_project_root() / "data"
        self.data_root = data_root
        self.snapshots_dir = self.data_root / "snapshots"
        self.manifest_path = self.data_root / "manifest.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        series_dict: Dict[str, pd.DataFrame],
        metadata_dict: Dict[str, SeriesMetadata],
        start_date: str,
        end_date: str,
        snapshot_tag: str = "v1.0",
    ) -> DataSnapshotMetadata:
        """
        Persist an immutable snapshot of all ingested time series.
        """
        timestamp = datetime.utcnow()
        hasher = hashlib.sha256()
        total_records = 0
        updated_metadata: Dict[str, SeriesMetadata] = {}

        # Pre-calculate deterministic hash
        for series_id in sorted(series_dict.keys()):
            df = series_dict[series_id]
            total_records += len(df)
            csv_bytes = df.to_csv(index=True).encode("utf-8")
            series_hash = hashlib.sha256(csv_bytes).hexdigest()
            hasher.update(csv_bytes)
            
            meta = metadata_dict.get(series_id)
            if meta:
                meta.file_sha256 = series_hash
                meta.record_count = len(df)
                updated_metadata[series_id] = meta

        master_hash = hasher.hexdigest()
        snapshot_id = f"snap_{snapshot_tag}_{master_hash[:12]}"
        snapshot_dir = self.snapshots_dir / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save files
        for series_id, df in series_dict.items():
            file_path = snapshot_dir / f"{series_id}.parquet"
            # Fallback to csv if pyarrow is not installed
            try:
                df.to_parquet(file_path)
            except Exception:
                file_path = snapshot_dir / f"{series_id}.csv"
                df.to_csv(file_path)

        snapshot_meta = DataSnapshotMetadata(
            snapshot_id=snapshot_id,
            creation_timestamp=timestamp,
            start_date=start_date,
            end_date=end_date,
            series_manifest=updated_metadata,
            sha256_checksum=master_hash,
            total_records=total_records,
        )

        # Write snapshot metadata
        with open(snapshot_dir / "snapshot_metadata.json", "w", encoding="utf-8") as f:
            json.dump(snapshot_meta.to_dict(), f, indent=2)

        # Update root manifest.json
        self._update_manifest(snapshot_meta)

        return snapshot_meta

    def _update_manifest(self, snapshot_meta: DataSnapshotMetadata) -> None:
        manifest_data: Dict[str, Any] = {"snapshots": {}}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                pass

        manifest_data["latest_snapshot"] = snapshot_meta.snapshot_id
        manifest_data["snapshots"][snapshot_meta.snapshot_id] = snapshot_meta.to_dict()

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def load_snapshot(self, snapshot_id: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Load an immutable snapshot by ID.
        """
        snapshot_dir = self.snapshots_dir / snapshot_id
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

        meta_file = snapshot_dir / "snapshot_metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        data = {}
        for series_id in meta_dict.get("series", {}).keys():
            parquet_path = snapshot_dir / f"{series_id}.parquet"
            csv_path = snapshot_dir / f"{series_id}.csv"
            if parquet_path.exists():
                data[series_id] = pd.read_parquet(parquet_path)
            elif csv_path.exists():
                data[series_id] = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        return data, meta_dict

    def verify_snapshot_integrity(self, snapshot_id: str) -> bool:
        """
        Verify cryptographic SHA-256 checksum against manifest.
        """
        snapshot_dir = self.snapshots_dir / snapshot_id
        meta_file = snapshot_dir / "snapshot_metadata.json"
        if not meta_file.exists():
            return False

        with open(meta_file, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        expected_hash = meta_dict["sha256_checksum"]
        hasher = hashlib.sha256()

        for series_id in sorted(meta_dict.get("series", {}).keys()):
            parquet_path = snapshot_dir / f"{series_id}.parquet"
            csv_path = snapshot_dir / f"{series_id}.csv"
            try:
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                elif csv_path.exists():
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                else:
                    return False
            except Exception:
                # Corrupted file detected
                return False
            hasher.update(df.to_csv(index=True).encode("utf-8"))

        return hasher.hexdigest() == expected_hash
