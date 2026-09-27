"""
Unit tests verifying all configuration files are present and well-formed.
"""

import pytest

from src.utils.config import get_project_root, load_all_configs, load_config


@pytest.mark.unit
def test_load_all_configs() -> None:
    configs = load_all_configs()
    required_keys = ["data", "models", "validation", "ensemble", "calibration", "online", "backtest"]
    for key in required_keys:
        assert key in configs, f"Expected configuration '{key}' to be loaded."
        assert isinstance(configs[key], dict)


@pytest.mark.unit
def test_data_config_structure() -> None:
    cfg = load_config("data")
    assert "universe" in cfg
    assert "equities" in cfg["universe"]
    assert "nifty_50" in cfg["universe"]["equities"]
    assert "missing_data_policy" in cfg


@pytest.mark.unit
def test_models_config_structure() -> None:
    cfg = load_config("models")
    assert cfg["n_regimes"] == 5
    assert len(cfg["regime_names"]) == 5
    assert "frequentist_hmm" in cfg
    assert "bayesian_hmm" in cfg
    assert "foundation_models" in cfg


@pytest.mark.unit
def test_validation_config_splits() -> None:
    cfg = load_config("validation")
    assert "splits" in cfg
    assert "training" in cfg["splits"]
    assert "calibration" in cfg["splits"]
    assert "test" in cfg["splits"]
    assert cfg["splits"]["training"]["end_date"] < cfg["splits"]["calibration"]["start_date"]
    assert cfg["splits"]["calibration"]["end_date"] < cfg["splits"]["test"]["start_date"]
