from pathlib import Path

from nanoresearch.config.paths import (
    get_bridge_install_dir,
    get_cli_history_path,
    get_cron_dir,
    get_data_dir,
    get_legacy_sessions_dir,
    get_logs_dir,
    get_media_dir,
    get_runtime_subdir,
    get_workspace_path,
    is_default_workspace,
)


def test_runtime_dirs_follow_config_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-a" / "config.json"
    monkeypatch.setattr("nanoresearch.config.paths.get_config_path", lambda: config_file)

    assert get_data_dir() == config_file.parent
    assert get_runtime_subdir("cron") == config_file.parent / "cron"
    assert get_cron_dir() == config_file.parent / "cron"
    assert get_logs_dir() == config_file.parent / "logs"


def test_media_dir_supports_channel_namespace(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-b" / "config.json"
    monkeypatch.setattr("nanoresearch.config.paths.get_config_path", lambda: config_file)

    assert get_media_dir() == config_file.parent / "media"
    assert get_media_dir("telegram") == config_file.parent / "media" / "telegram"


def test_shared_and_legacy_paths_remain_global(monkeypatch):
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    assert get_cli_history_path() == Path.home() / ".nanoresearch" / "history" / "cli_history"
    assert get_bridge_install_dir() == Path.home() / ".nanoresearch" / "bridge"
    assert get_legacy_sessions_dir() == Path.home() / ".nanoresearch" / "sessions"


def test_shared_and_legacy_paths_follow_nanoresearch_home(monkeypatch, tmp_path):
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "tenant_y"))
    assert get_cli_history_path() == tmp_path / "tenant_y" / "history" / "cli_history"
    assert get_bridge_install_dir() == tmp_path / "tenant_y" / "bridge"
    assert get_legacy_sessions_dir() == tmp_path / "tenant_y" / "sessions"


def test_workspace_path_is_explicitly_resolved(monkeypatch):
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    assert get_workspace_path() == Path.home() / ".nanoresearch" / "workspace"
    assert get_workspace_path("~/custom-workspace") == Path.home() / "custom-workspace"


def test_workspace_path_follows_nanoresearch_home(monkeypatch, tmp_path):
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "tenant_z"))
    assert get_workspace_path() == tmp_path / "tenant_z" / "workspace"


def test_is_default_workspace_distinguishes_default_and_custom_paths(monkeypatch):
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    assert is_default_workspace(None) is True
    assert is_default_workspace(Path.home() / ".nanoresearch" / "workspace") is True
    assert is_default_workspace("~/custom-workspace") is False


def test_get_nanoresearch_home_defaults_to_user_home(monkeypatch):
    from nanoresearch.config.loader import get_nanoresearch_home

    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_HOME", raising=False)

    assert get_nanoresearch_home() == Path.home() / ".nanoresearch"


def test_get_nanoresearch_home_respects_env_override(monkeypatch, tmp_path):
    from nanoresearch.config.loader import get_nanoresearch_home

    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "tenant_x"))

    assert get_nanoresearch_home() == tmp_path / "tenant_x"


def test_get_nanoresearch_home_expands_tilde(monkeypatch):
    from nanoresearch.config.loader import get_nanoresearch_home

    monkeypatch.delenv("NANOBOT_HOME", raising=False)
    monkeypatch.setenv("NANORESEARCH_HOME", "~/custom-nr-root")

    assert get_nanoresearch_home() == Path.home() / "custom-nr-root"


def test_nanobot_home_legacy_env_is_translated(monkeypatch, tmp_path, recwarn):
    from nanoresearch.config.loader import get_nanoresearch_home
    from nanoresearch.utils.env_compat import _reset_for_tests

    _reset_for_tests()
    monkeypatch.delenv("NANORESEARCH_HOME", raising=False)
    monkeypatch.setenv("NANOBOT_HOME", str(tmp_path / "legacy_tenant"))

    result = get_nanoresearch_home()

    assert result == tmp_path / "legacy_tenant"
    deprecation_msgs = [
        str(w.message) for w in recwarn.list
        if issubclass(w.category, DeprecationWarning) and "NANOBOT_HOME" in str(w.message)
    ]
    assert any("NANORESEARCH_HOME" in m for m in deprecation_msgs), (
        f"expected a DeprecationWarning naming both NANOBOT_HOME and NANORESEARCH_HOME, got {deprecation_msgs}"
    )
