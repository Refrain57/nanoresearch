"""Configuration loading utilities."""

import json
import os
from pathlib import Path

import pydantic
from loguru import logger

from nanoresearch.config.schema import Config
from nanoresearch.utils.env_compat import apply_legacy_env_compat

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None


def get_nanoresearch_home() -> Path:
    """Return the base directory for all NanoResearch runtime state.

    Reads ``NANORESEARCH_HOME``; falls back to ``~/.nanoresearch``. Calls the
    legacy env-compat shim first so ``NANOBOT_HOME`` is honoured for scripts
    that bypass the main entrypoints. Does not create the directory.
    """
    apply_legacy_env_compat()
    raw = os.environ.get("NANORESEARCH_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".nanoresearch"


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return get_nanoresearch_home() / "config.json"


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            logger.warning("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json", by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data
