"""
Configuration Management Utility.

Loads, merges, and validates YAML configuration files from the config/ directory.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file by name (e.g. 'data.yaml' or 'data').
    """
    if config_dir is None:
        config_dir = get_project_root() / "config"

    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = f"{config_name}.yaml"

    config_path = config_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        content: Dict[str, Any] = yaml.safe_load(f) or {}

    return content


def load_all_configs(config_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load all standard configuration files into a unified dictionary."""
    config_names = [
        "data",
        "models",
        "validation",
        "ensemble",
        "calibration",
        "online",
        "backtest",
        "logging",
    ]
    bundle = {}
    for name in config_names:
        try:
            bundle[name] = load_config(name, config_dir)
        except FileNotFoundError:
            pass
    return bundle
