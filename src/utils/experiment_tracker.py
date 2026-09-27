"""
Experiment Tracking & Registry Management.

Maintains immutable records of all model runs, parameters, priors, MCMC diagnostics,
and validation outcomes in experiments/registry/ adhering to EXPERIMENT_PROTOCOL.md.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import get_project_root
from src.utils.reproducibility import get_git_commit_hash


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    data_snapshot_id: str
    feature_version: str
    model_family: str
    model_version: str
    training_period: Dict[str, str]
    calibration_period: Dict[str, str]
    test_period: Dict[str, str]
    random_seed: int
    hyperparameters: Dict[str, Any]
    priors: Dict[str, Any]
    metrics: Dict[str, Any]
    diagnostics: Dict[str, Any]
    calibration_results: Dict[str, Any]
    output_artifacts: Dict[str, str]


class ExperimentTracker:
    """Manages recording, indexing, and querying of experiment runs."""

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        if registry_dir is None:
            registry_dir = get_project_root() / "experiments" / "registry"
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "index.json"
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self.index_path.exists():
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({"experiments": []}, f, indent=2)

    def log_experiment(self, record: ExperimentRecord) -> Path:
        """Persist experiment record and update the registry catalog."""
        record_dict = asdict(record)
        record_file = self.registry_dir / f"{record.experiment_id}.json"

        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(record_dict, f, indent=2)

        # Update index
        with open(self.index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        # Append or replace
        experiments = [e for e in index_data.get("experiments", []) if e["experiment_id"] != record.experiment_id]
        experiments.append(
            {
                "experiment_id": record.experiment_id,
                "timestamp": record.timestamp,
                "model_family": record.model_family,
                "model_version": record.model_version,
                "git_commit": record.git_commit,
                "metrics": record.metrics,
                "file_path": str(record_file.relative_to(get_project_root())),
            }
        )
        index_data["experiments"] = experiments

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        return record_file

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        record_file = self.registry_dir / f"{experiment_id}.json"
        if not record_file.exists():
            return None
        with open(record_file, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
            return data

    def list_experiments(self) -> List[Dict[str, Any]]:
        with open(self.index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
            experiments: List[Dict[str, Any]] = index_data.get("experiments", [])
            return experiments
