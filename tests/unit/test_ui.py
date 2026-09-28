"""
Unit Test for Streamlit UI App Navigation and Page Rendering.

Uses streamlit.testing.v1.AppTest to execute and verify all 8 pages:
- Live Regime Monitor
- Why This Regime?
- Model Lab
- Market Replay
- Tactical Allocation & Risk
- Model Governance
- Certified Audit Trail
- Regime Arena
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_ui_data_dependencies_exist():
    feats_p = Path("data/processed/features_matrix.parquet")
    preds_p = Path("data/processed/calibrated_ensemble_predictions.parquet")
    models_p = Path("data/processed/model_predictions_matrix.parquet")

    assert feats_p.exists(), "Features matrix exists for Streamlit UI"
    assert preds_p.exists(), "Calibrated predictions exist for Streamlit UI"
    assert models_p.exists(), "Model predictions exist for Streamlit UI"


def test_streamlit_all_pages_render_without_exceptions():
    pages = [
        "1. Live Regime Monitor",
        "2. Why This Regime? (SHAP)",
        "3. Model Lab",
        "4. Historical Market Replay",
        "5. Risk & Tactical Allocation",
        "6. Model Governance",
        "7. Point-in-Time Audit Trail",
        "8. Regime Arena (Simulation)",
    ]

    app_path = (Path(__file__).parents[2] / "src" / "ui" / "app.py").resolve()

    for page_name in pages:
        at = AppTest.from_file(str(app_path), default_timeout=20)
        at.run()
        # Find sidebar radio and select page
        if len(at.sidebar.radio) > 0:
            at.sidebar.radio[0].set_value(page_name)
            at.run()
        assert len(at.exception) == 0, f"Page {page_name} raised exception: {at.exception}"
