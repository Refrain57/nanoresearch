"""Configuration loading utilities."""

import json
import os
from pathlib import Path
from typing import Literal

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


_VALID_MODES = ("server", "local")


def get_mode() -> Literal["server", "local"]:
    """Return the deployment mode (NANORESEARCH_MODE env var, default 'local').

    server: 凭证唯一来源 = user_settings.extra.providers；config.json / env 兜底全关
    local:  沿用 user_providers > config.json > settings.yaml > env 链路（本地 dev）
    """
    raw = os.environ.get("NANORESEARCH_MODE", "local").lower()
    if raw not in _VALID_MODES:
        raise ValueError(
            f"NANORESEARCH_MODE must be one of {_VALID_MODES}, got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def env_key_fallback_allowed() -> bool:
    """True iff env-var API key fallback is permitted (i.e. local mode)."""
    return get_mode() == "local"


def env_key_or_raise(env_name: str, *, role: str) -> str:
    """Read env var in local mode; raise in server mode or when var unset.

    Used as fallback when user_settings + config.json have no key. In server
    mode this short-circuits to raise — server deployments must not silently
    spend host platform credentials.
    """
    if get_mode() == "server":
        raise RuntimeError(
            f"API key for role '{role}' must come from user_settings.extra.providers "
            f"in server mode (NANORESEARCH_MODE=server); env var {env_name} fallback disabled."
        )
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(
            f"{env_name} not set (role={role!r}); set the env var or pass api_key explicitly."
        )
    return value


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
