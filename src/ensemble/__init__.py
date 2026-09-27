"""
Multi-Model Ensembling and Uncertainty Aggregation package.
"""

from src.ensemble.base import BaseEnsemble
from src.ensemble.bma import BayesianModelAveraging, ConstrainedStackingEnsemble

__all__ = [
    "BaseEnsemble",
    "BayesianModelAveraging",
    "ConstrainedStackingEnsemble",
]
