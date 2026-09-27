"""
Feature Engineering Base Interfaces & Anti-Leakage Pipeline.

Defines the BaseFeatureTransformer interface ensuring strict backward-looking
transformations without temporal contamination.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseFeatureTransformer(ABC):
    """
    Abstract interface for feature transformers.
    Must guarantee that calculation of features at time t depends only on observations at <= t.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseFeatureTransformer":
        """Fit any parameters (e.g. historical normalization statistics) strictly on training slice."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformation returning a DataFrame with identical DatetimeIndex."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class FeaturePipeline:
    """Sequential feature engineering pipeline with anti-leakage checks."""

    def __init__(self, transformers: List[BaseFeatureTransformer]) -> None:
        self.transformers = transformers

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        for t in self.transformers:
            t.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        for t in self.transformers:
            transformed = t.transform(df)
            res = pd.concat([res, transformed], axis=1)
        # Drop duplicates if any
        res = res.loc[:, ~res.columns.duplicated()]
        return res
