"""
Entry point for running nanobot as a module: python -m nanobot
"""

from nanobot.utils.env_compat import apply_legacy_env_compat

apply_legacy_env_compat()

from nanobot.cli.commands import app  # noqa: E402

if __name__ == "__main__":
    app()
