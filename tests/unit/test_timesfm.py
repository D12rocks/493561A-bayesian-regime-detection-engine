"""
Unit test for TimesFM Foundation Model Adapter.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.foundation.timesfm_adapter import TimesFMRegimeAdapter


def test_timesfm_regime_adapter_fit_and_predict():
    rng = np.random.RandomState(42)
    n_obs = 80
    df = pd.DataFrame(rng.randn(n_obs, 2), columns=["ret", "vol"])

    adapter = TimesFMRegimeAdapter(
        model_id="google/timesfm-2.0",
        context_length=32,
        patch_len=8,
        embedding_dim=16,
        random_seed=42,
    )
    adapter.fit(df)
    assert adapter.is_fitted

    diag = adapter.diagnostics()
    assert "TimesFM" in diag["architecture"]
    assert "regime_separation" in diag
    assert "silhouette_score" in diag["regime_separation"]

    probs = adapter.predict_proba(df)
    assert probs.shape == (n_obs, 5)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
