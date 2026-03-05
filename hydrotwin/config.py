"""Configuration loader for HydroTwin OS."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_CONFIG_CACHE: dict[str, Any] | None = None
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return pattern.sub(_replace, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration with env-var resolution."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _CONFIG_CACHE = _resolve_env_vars(raw)
    return _CONFIG_CACHE


def get_config_section(section: str, config_path: str | Path | None = None) -> dict[str, Any]:
    """Get a specific section from the configuration."""
    cfg = load_config(config_path)
    if section not in cfg:
        raise KeyError(f"Config section '{section}' not found. Available: {list(cfg.keys())}")
    return cfg[section]


def reset_config_cache() -> None:
    """Reset the config cache (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
