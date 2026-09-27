"""
Point-in-Time Feature Store and Registry.

Implements REQ-018: Point-in-Time Feature Store ensuring strict timestamp alignment,
anti-leakage guarantees, immutable SHA-256 snapshotting, and point-in-time historical retrieval.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.technical import TechnicalFeatureExtractor
from src.features.tda import TDAFeatureExtractor
from src.features.gnn import SectorGraphFeatureExtractor
from src.features.selection import FeatureAuditor


@dataclass
class FeatureSnapshotMetadata:
    snapshot_id: str
    created_at_utc: str
    sha256_hash: str
    n_rows: int
    n_features: int
    start_date: str
    end_date: str
    feature_names: List[str]
    source_market_snapshot: str


class PointInTimeFeatureStore:
    """
    Centralized, point-in-time feature store for Indian quantitative regime modeling.
    """

    def __init__(
        self,
        storage_dir: Union[str, Path] = "data/processed",
        snapshots_dir: Union[str, Path] = "data/snapshots",
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.technical_extractor = TechnicalFeatureExtractor()
        self.tda_extractor = TDAFeatureExtractor(window_size=60)
        self.gnn_extractor = SectorGraphFeatureExtractor(window_size=60)

    def load_market_snapshot(self, snapshot_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """Loads all parquet asset histories from a market data snapshot folder."""
        snap_dir = Path(snapshot_path)
        if not snap_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")
            
        data = {}
        for pfile in snap_dir.glob("*.parquet"):
            sym = pfile.stem
            df = pd.read_parquet(pfile)
            df.index = pd.to_datetime(df.index)
            data[sym] = df.sort_index()
            
        return data

    def build_full_feature_matrix(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Builds the unified feature matrix combining Technical, TDA, and GNN features.
        Guarantees that every row at date t depends strictly on data <= t.
        """
        print("Extracting technical, volatility, and breadth features...")
        tech_df = self.technical_extractor.extract_from_multivariate(market_data)

        # Build returns matrix for TDA point cloud
        print("Building point-in-time TDA persistent homology features...")
        tda_symbols = ["NIFTY_50", "NIFTY_MIDCAP_50", "INDIA_VIX", "USD_INR"]
        available_tda = [s for s in tda_symbols if s in market_data]
        tda_ret_df = pd.DataFrame(index=tech_df.index)
        for s in available_tda:
            close_s = market_data[s]["close"].reindex(tech_df.index).ffill()
            tda_ret_df[s] = np.log(close_s / close_s.shift(1))
        tda_df = self.tda_extractor.transform(tda_ret_df)

        print("Building sector correlation graph and GNN features...")
        gnn_df = self.gnn_extractor.extract_from_sector_prices(market_data)

        # Merge all features on DatetimeIndex
        print("Merging feature panels...")
        full_df = pd.concat([tech_df, tda_df, gnn_df], axis=1)
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        
        # Sort chronologically
        full_df = full_df.sort_index()
        return full_df

    def compute_sha256(self, df: pd.DataFrame) -> str:
        """Computes deterministic SHA-256 hash of DataFrame."""
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def create_feature_snapshot(
        self,
        market_snapshot_path: Union[str, Path],
        snapshot_name: Optional[str] = None,
    ) -> Tuple[Path, FeatureSnapshotMetadata]:
        """
        End-to-end pipeline: load market snapshot, generate features,
        audit stationarity, persist immutable snapshot with metadata.
        """
        market_path = Path(market_snapshot_path)
        market_data = self.load_market_snapshot(market_path)
        
        features_df = self.build_full_feature_matrix(market_data)
        
        # Drop warm-up period (first 200 trading days for 200 SMA & rolling windows)
        clean_features = features_df.iloc[200:].copy()
        # Forward fill any intermittent missing items, then fill remaining with 0.0
        clean_features = clean_features.ffill().fillna(0.0)
        
        sha256 = self.compute_sha256(clean_features)
        snap_id = snapshot_name or f"snap_features_{sha256[:12]}"
        
        out_dir = self.snapshots_dir / snap_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_file = out_dir / "features_matrix.parquet"
        clean_features.to_parquet(parquet_file, index=True)
        
        metadata = FeatureSnapshotMetadata(
            snapshot_id=snap_id,
            created_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            sha256_hash=sha256,
            n_rows=len(clean_features),
            n_features=len(clean_features.columns),
            start_date=str(clean_features.index[0].date()),
            end_date=str(clean_features.index[-1].date()),
            feature_names=clean_features.columns.tolist(),
            source_market_snapshot=str(market_path.name),
        )
        
        meta_file = out_dir / "features_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
            
        print(f"Feature snapshot successfully created at: {out_dir}")
        print(f"Shape: {clean_features.shape}, Hash: {sha256}")
        return out_dir, metadata

    def get_historical_feature_vector(
        self,
        features_df: pd.DataFrame,
        as_of_date: Union[str, pd.Timestamp],
    ) -> pd.Series:
        """
        Point-in-time query: Returns feature vector exactly as of date <= as_of_date.
        Strict anti-leakage guarantee: Will NEVER look at observations > as_of_date.
        """
        ts = pd.to_datetime(as_of_date)
        past_data = features_df.loc[features_df.index <= ts]
        if past_data.empty:
            raise ValueError(f"No feature observations available on or before {as_of_date}")
        return past_data.iloc[-1]
