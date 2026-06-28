"""One-time migration: copy settings.yaml LLM api_keys into config.json providers.

Before removing the redundant ``api_key`` fields from settings.yaml, this module
ensures no user loses access: for each key found in settings.yaml, if the matching
provider in config.json has no key yet, it copies the key over.

Usage
-----
    from nanoresearch.config.migration import migrate_llm_keys

    report = migrate_llm_keys(dry_run=True)
    print(report)  # review what would change
    migrate_llm_keys(dry_run=False)  # apply
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from nanoresearch.config.loader import get_config_path, set_config_path

logger = logging.getLogger(__name__)

# Provider name mappings: settings.yaml provider name → config.json provider field name.
# Most use the same name, but a few differ (e.g. "openai-compat" → "custom").
# The key is the normalized provider string from settings.yaml;
# the value is the attribute name on ProvidersConfig.
_PROVIDER_FIELD_OVERRIDES: dict[str, str] = {
    "openai_compat": "custom",
}


def _normalise_provider(name: str) -> str:
    """Normalise a provider string to the form used in config.json ProvidersConfig."""
    raw = name.lower().replace("-", "_")
    return _PROVIDER_FIELD_OVERRIDES.get(raw, raw)


# Sections in settings.yaml that carry an api_key we care about.
# Each entry: (yaml_path_prefix, yaml_provider_field, yaml_key_field, config_provider_field_suffix)
_SECTIONS = (
    ("llm", "provider", "api_key"),
    ("embedding", "provider", "api_key"),
    ("vision_llm", "provider", "api_key"),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def migrate_llm_keys(
    settings_path: str | Path | None = None,
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy settings.yaml LLM api_keys into config.json providers.

    Parameters
    ----------
    settings_path:
        Path to settings.yaml.  Defaults to ``~/.nanoresearch/settings.yaml``.
    config_path:
        Path to config.json.  Defaults to ``~/.nanoresearch/config.json``.
    dry_run:
        If True, report what would change without writing anything.

    Returns
    -------
    dict with keys:
        - migrated: list of ``{section, provider, key_hint}``
        - skipped_already_exists: list of ``{section, provider}``
        - skipped_no_provider: list of ``{section, provider}``
        - errors: list of ``{section, message}``
    """
    from nanoresearch.rag.core.settings import default_settings_path

    # --- resolve paths ---
    settings_path = settings_path or default_settings_path()
    config_path = config_path or get_config_path()

    report: dict[str, Any] = {
        "migrated": [],
        "skipped_already_exists": [],
        "skipped_no_provider": [],
        "errors": [],
    }

    # --- load settings.yaml ---
    settings_path = Path(settings_path)
    if not settings_path.exists():
        logger.info("migrate_llm_keys: settings.yaml not found at %s, skipping", settings_path)
        return report

    with settings_path.open("r", encoding="utf-8") as f:
        settings_data = yaml.safe_load(f) or {}

    # --- load config.json ---
    config_path = Path(config_path)
    if not config_path.exists():
        logger.info("migrate_llm_keys: config.json not found at %s, skipping", config_path)
        return report

    with config_path.open("r", encoding="utf-8") as f:
        config_data: dict = json.load(f)

    providers = config_data.setdefault("providers", {})
    changed = False

    for section, provider_field, key_field in _SECTIONS:
        sec = settings_data.get(section)
        if not isinstance(sec, dict):
            continue

        api_key = sec.get(key_field)
        if not api_key:
            continue  # no key in settings.yaml for this section

        raw_provider = sec.get(provider_field, "")
        if not raw_provider:
            report["errors"].append({
                "section": section,
                "message": f"settings.yaml.{section}.{provider_field} is empty, cannot determine target provider",
            })
            continue

        cfg_field = _normalise_provider(raw_provider)

        if cfg_field not in providers:
            report["skipped_no_provider"].append({"section": section, "provider": cfg_field})
            logger.warning(
                "migrate_llm_keys: %s provider=%r not found in config.json providers, "
                "skipping key %s",
                section, cfg_field, _mask_key(api_key),
            )
            continue

        cfg_provider = providers[cfg_field]
        if not isinstance(cfg_provider, dict):
            cfg_provider = {}
            providers[cfg_field] = cfg_provider

        existing = cfg_provider.get("apiKey") or cfg_provider.get("api_key") or ""

        if existing and existing != api_key:
            report["skipped_already_exists"].append({"section": section, "provider": cfg_field})
            logger.info(
                "migrate_llm_keys: %s → config.json.providers.%s already has api_key, skipping",
                section, cfg_field,
            )
            continue

        if existing == api_key:
            # Already in sync — nothing to do
            report["skipped_already_exists"].append({"section": section, "provider": cfg_field})
            continue

        # --- copy the key ---
        cfg_provider["apiKey"] = api_key
        changed = True
        report["migrated"].append({
            "section": section,
            "provider": cfg_field,
            "key_hint": _mask_key(api_key),
        })
        logger.info(
            "migrate_llm_keys: %s → config.json.providers.%s  key=%s",
            section, cfg_field, _mask_key(api_key),
        )

    if changed and not dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        logger.info("migrate_llm_keys: config.json updated")

    if changed and dry_run:
        logger.info("migrate_llm_keys: dry-run mode — config.json NOT written")

    return report


def _mask_key(key: str) -> str:
    """Show first 4 + last 4 chars, mask the rest."""
    if not key:
        return ""
    if len(key) <= 12:
        return key[:4] + "..." + key[-4:] if len(key) > 8 else key
    return key[:4] + "..." + key[-4:]