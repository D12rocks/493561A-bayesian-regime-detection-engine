"""
Orchestration Pipeline Runners.

Coordinates end-to-end data ingestion, feature generation, model training,
ensembling, calibration, and prediction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from src.models.contracts import RegimePrediction


class BasePipelineRunner(ABC):
    """Abstract orchestrator for end-to-end workflows."""

    @abstractmethod
    def run_batch_training(self, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute full historical train -> calibrate -> ensemble pipeline."""
        pass

    @abstractmethod
    def run_inference(self, as_of_date: Optional[str] = None) -> RegimePrediction:
        """Execute point-in-time regime inference and conformal set generation."""
        pass
