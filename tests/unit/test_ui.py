"""
Unit Test for UI App Compilation and Data Ingestion.
"""

from pathlib import Path
import pandas as pd
import pytest


def test_ui_data_dependencies_exist():
    feats_p = Path("data/processed/features_matrix.parquet")
    preds_p = Path("data/processed/calibrated_ensemble_predictions.parquet")
    models_p = Path("data/processed/model_predictions_matrix.parquet")

    assert feats_p.exists(), "Features matrix exists for Streamlit UI"
    assert preds_p.exists(), "Calibrated predictions exist for Streamlit UI"
    assert models_p.exists(), "Model predictions exist for Streamlit UI"

    df = pd.read_parquet(preds_p)
    assert len(df) > 1000
    assert "ensemble_prob_risk_on" in df.columns
