"""
Structured Logging Utility.

Configures application-wide logging adhering to config/logging.yaml.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional
import yaml

from src.utils.config import get_project_root


def setup_logging(config_path: Optional[Path] = None, default_level: int = logging.INFO) -> None:
    """Initialize structured logging from config or fallback."""
    if config_path is None:
        config_path = get_project_root() / "config" / "logging.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def get_logger(name: str) -> logging.Logger:
    """Get named logger instance."""
    return logging.getLogger(name)
