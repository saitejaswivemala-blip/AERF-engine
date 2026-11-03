"""
logging_config.py

One place to configure logging so every module gets consistent formatting
and (per settings.yaml) optionally writes to a log file as well as the
console. Call get_logger(__name__) at the top of any module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.config_loader import get_settings, PROJECT_ROOT

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_cfg = settings.get("logging", {})
    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger("aerf")
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional file handler
    if log_cfg.get("log_to_file", True):
        log_dir_setting = settings.get("paths", {}).get("log_dir", "logs")
        log_dir_path = Path(log_dir_setting)
        if not log_dir_path.is_absolute():
            log_dir_path = PROJECT_ROOT / log_dir_path
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file = log_dir_path / log_cfg.get("log_filename", "aerf_engine.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger under the shared 'aerf' namespace."""
    _configure_root_logger()
    return logging.getLogger(f"aerf.{name}")
