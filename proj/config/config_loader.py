"""
config_loader.py

Single point of access for settings.yaml. Every other module reads
configuration through get_settings() rather than opening the YAML file
itself, so there is exactly one place that knows the file's location
and structure.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (config/config_loader.py -> aerf-engine/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class ConfigError(Exception):
    """Raised when settings.yaml is missing or malformed."""


@functools.lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    """
    Load and cache settings.yaml. Cached because it is read frequently
    (every module that needs a path or a threshold calls this) and the
    file does not change during a single run.
    """
    if not SETTINGS_PATH.exists():
        raise ConfigError(
            f"settings.yaml not found at {SETTINGS_PATH}. "
            "Run this project from within aerf-engine/ or check your install."
        )
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"settings.yaml is not valid YAML: {exc}") from exc

    if not data:
        raise ConfigError("settings.yaml is empty.")

    return data


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path from settings.yaml relative to the project root,
    creating parent directories if they don't exist yet (safe for
    output paths like the database file or log directory).
    """
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
