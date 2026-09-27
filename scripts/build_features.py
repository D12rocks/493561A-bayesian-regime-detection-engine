"""
Build Point-in-Time Feature Snapshot and Stationarity Diagnostics.

Executes end-to-end feature extraction over the real Indian market data snapshot,
runs ADF stationarity and VIF multicollinearity audits, and serializes
the official feature snapshot with metadata.
"""

from pathlib import Path
import pandas as pd
from src.features.store import PointInTimeFeatureStore
from src.features.selection import FeatureAuditor


def main() -> None:
    market_snap_path = Path("data/snapshots/snap_phase1_market_data_c6c46ed7b46c")
    if not market_snap_path.exists():
        raise FileNotFoundError(f"Market snapshot {market_snap_path} not found.")

    store = PointInTimeFeatureStore(
        storage_dir="data/processed",
        snapshots_dir="data/snapshots",
    )

    print("=== Launching Point-in-Time Feature Engineering Pipeline ===")
    out_dir, metadata = store.create_feature_snapshot(
        market_snapshot_path=market_snap_path,
        snapshot_name="snap_phase2_features_v1",
    )

    parquet_path = out_dir / "features_matrix.parquet"
    features_df = pd.read_parquet(parquet_path)
    print(f"Features matrix shape: {features_df.shape}")
    print(f"Features: {list(features_df.columns)}")

    print("\n=== Running Feature Stationarity (ADF) Audit ===")
    stationarity_df = FeatureAuditor.test_stationarity(features_df)
    
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    stationarity_csv = tables_dir / "feature_stationarity_summary.csv"
    stationarity_df.to_csv(stationarity_csv, index=False)
    print(f"Stationarity summary saved to: {stationarity_csv}")
    print(stationarity_df.to_string())

    print("\n=== Running VIF Multicollinearity Audit ===")
    vif_df = FeatureAuditor.compute_vif(features_df)
    vif_csv = tables_dir / "feature_vif_summary.csv"
    vif_df.to_csv(vif_csv, index=False)
    print(f"VIF summary saved to: {vif_csv}")
    print(vif_df.head(10).to_string())

    # Save to processed directory for live engine access
    features_df.to_parquet("data/processed/features_matrix.parquet")
    print("Features matrix also saved to data/processed/features_matrix.parquet")


if __name__ == "__main__":
    main()
